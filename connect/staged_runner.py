"""Independent, disk-cached PathGen / MedGemma / MUSK Slurm stages."""
import csv
import hashlib
import json
import re
import time
import uuid
from pathlib import Path

from notebook_runner import Settings, find_connect, _command, _diagnostic
from integrated_report import render_report


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _stamp(path):
    """Track model/environment file metadata without reading GB of weights on login."""
    path = Path(path).resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        files = sorted(p for p in path.rglob("*") if p.is_file() and ".cache" not in p.parts)
    else:
        files = [path]
    return [[str(p), p.stat().st_size, p.stat().st_mtime_ns] for p in files]


def _write(path, record):
    path = Path(path)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


class ReportSession:
    def __init__(self, filename, settings=None):
        self.settings = settings or Settings(find_connect())
        if self.settings.poll_seconds <= 0 or self.settings.timeout_seconds <= 0:
            raise ValueError("Polling interval and timeout must be positive")
        filename = filename.strip()
        if not filename or "/" in filename or "\\" in filename or Path(filename).suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
            raise ValueError("Enter an image filename only, e.g. TUM-YHTSLLSL.tif")
        self.image = (self.settings.image_root / filename).resolve(strict=True)
        self.digest = _hash(self.image)
        identity = hashlib.sha256((str(self.image) + self.digest).encode()).hexdigest()[:24]
        self.root = self.settings.connect / "reports" / "staged" / identity
        self.root.mkdir(parents=True, exist_ok=True)
        self.logs = self.settings.connect / "logs"
        self.logs.mkdir(exist_ok=True)
        self.caption = self.explanation = self.prediction = None
        self.caption_path = self.explanation_path = self.prediction_path = None

    def _unchanged(self):
        if _hash(self.image) != self.digest:
            raise RuntimeError("The image changed. Start a new ReportSession.")

    def _validate(self, stage, artifact):
        self._unchanged()
        if stage == "musk":
            with artifact.open(encoding="utf-8", newline="") as stream:
                rows = list(csv.DictReader(stream))
            if len(rows) != 1 or Path(rows[0]["image_path"]).resolve() != self.image:
                raise ValueError("MUSK result image mismatch")
            render_report({"simplified_caption": "validation"}, rows[0])
            return rows[0]
        record = json.loads(artifact.read_text(encoding="utf-8"))
        if record.get("image_sha256") != self.digest or Path(record["image_path"]).resolve() != self.image:
            raise ValueError("Caption/explanation image mismatch")
        field = "caption" if stage == "pathgen" else "simplified_caption"
        if not isinstance(record.get(field), str) or not record[field].strip():
            raise ValueError(f"Empty {field}")
        return record

    def _wait(self, job):
        deadline = time.monotonic() + self.settings.timeout_seconds
        missing = None
        terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                    "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
        try:
            while time.monotonic() < deadline:
                result = _command(["sacct", "-n", "-X", "-P", "-j", job,
                                   "--format=JobIDRaw,State,ExitCode"], self.settings.connect)
                state = code = None
                if result.returncode == 0:
                    for line in result.stdout.splitlines():
                        fields = line.strip().split("|")
                        if len(fields) >= 3 and fields[0] == job:
                            state, code = fields[1].split()[0].rstrip("+"), fields[2]
                            break
                if state in terminal:
                    if state != "COMPLETED" or code != "0:0":
                        raise RuntimeError(f"Job {job}: {state} ({code})\n" + _diagnostic(job, self.logs))
                    return
                queue = _command(["squeue", "-h", "-j", job, "-o", "%T"], self.settings.connect)
                if queue.returncode == 0 and not queue.stdout.strip() and state is None:
                    missing = missing or time.monotonic()
                    if time.monotonic() - missing > 120:
                        raise RuntimeError(f"Cannot determine job {job} status. Check sacct before retrying.")
                else:
                    missing = None
                time.sleep(self.settings.poll_seconds)
            raise TimeoutError(f"Wait timed out: job {job}")
        except (KeyboardInterrupt, TimeoutError):
            cancelled = _command(["scancel", job], self.settings.connect)
            if cancelled.returncode:
                raise RuntimeError(f"Cancel job {job} manually using scancel.")
            raise

    def _stage(self, stage, signature, python, script, arguments, force=False, imported=None):
        self._unchanged()
        # Include local script content, settings, upstream output and model file metadata.
        signature.update(image_sha256=self.digest, image_path=str(self.image),
                         script_sha256=_hash(script), runner_sha256=_hash(Path(__file__)),
                         batch_sha256=_hash(self.settings.connect / "stage_delta.sbatch"))
        key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()
        folder = self.root / stage / key
        folder.mkdir(parents=True, exist_ok=True)
        manifest = folder / "state.json"
        lock = folder / "submit.lock"
        try:
            guard = lock.open("x")
        except FileExistsError:
            raise RuntimeError(f"Another submission is updating this stage. If its kernel crashed, inspect {lock}.")
        try:
            state = json.loads(manifest.read_text()) if manifest.exists() else {}
            if state.get("status") == "complete" and not force:
                artifact = Path(state["artifact"])
                if artifact.is_file() and _hash(artifact) == state["artifact_sha256"]:
                    return self._validate(stage, artifact), artifact
            if state.get("status") == "running":
                # Reconnect to the saved job, including after a notebook/kernel restart.
                if force:
                    raise RuntimeError(f"Job {state['job_id']} is still recorded as running; finish or cancel it first.")
            else:
                if stage == "medgemma" and imported is None:
                    check = _command([python, self.settings.connect / "check_medgemma.py",
                        "--base-model", self.settings.medgemma_base], self.settings.connect)
                    if check.returncode:
                        raise RuntimeError("MedGemma preflight failed; saved PathGen caption is unchanged:\n" + check.stderr)
                attempt = folder / uuid.uuid4().hex
                attempt.mkdir()
                artifact = attempt / ("musk.csv" if stage == "musk" else "result.json")
                if imported is not None:
                    self._validate(stage, imported)
                    artifact.write_bytes(imported.read_bytes())
                    state = {"status": "complete", "artifact": str(artifact),
                             "artifact_sha256": _hash(artifact), "imported_from": str(imported), "signature": signature}
                    _write(manifest, state)
                    return self._validate(stage, artifact), artifact
                command = ["sbatch", "--parsable", "--export=ALL", "--output", self.logs / "notebook-%j.out",
                    "--error", self.logs / "notebook-%j.err", self.settings.connect / "stage_delta.sbatch",
                    stage, python, script, *arguments, "--output", artifact]
                submitted = _command(command, self.settings.connect)
                if submitted.returncode:
                    raise RuntimeError("Slurm submission failed:\n" + submitted.stderr)
                job = submitted.stdout.strip().split(";")[0]
                if not re.fullmatch(r"\d+", job):
                    raise RuntimeError(f"Unexpected sbatch response: {submitted.stdout!r}")
                state = {"status": "running", "job_id": job, "artifact": str(artifact), "signature": signature}
                _write(manifest, state)
        finally:
            guard.close()
            lock.unlink()
        try:
            self._wait(state["job_id"])
            artifact = Path(state["artifact"])
            record = self._validate(stage, artifact)
        except (KeyboardInterrupt, TimeoutError):
            state["status"] = "interrupted"
            _write(manifest, state)
            raise
        except Exception:
            # Failed terminal jobs can be retried; transient monitoring errors retain job ID.
            status = _command(["sacct", "-n", "-X", "-P", "-j", state["job_id"],
                               "--format=JobIDRaw,State,ExitCode"], self.settings.connect)
            if status.returncode == 0:
                for line in status.stdout.splitlines():
                    parts = line.strip().split("|")
                    if len(parts) >= 3 and parts[0] == state["job_id"] and parts[1].split()[0].rstrip("+") in {"FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "COMPLETED", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}:
                        state["status"] = "failed"
                        _write(manifest, state)
            raise
        state.update(status="complete", artifact_sha256=_hash(artifact))
        _write(manifest, state)
        return record, artifact

    def pathgen(self, force=False, existing_caption=None):
        s = self.settings
        script = s.connect / "caption_pathgen.py"
        signature = {"model": _stamp(s.pathgen_model), "python": _stamp(s.pathgen_python),
                     "conv_mode": "llava_v1"}
        imported = Path(existing_caption).resolve(strict=True) if existing_caption else None
        if imported:
            record = self._validate("pathgen", imported)
            if str(Path(record["caption_model"]).resolve()) != str(s.pathgen_model.resolve()) or record.get("conv_mode") != "llava_v1":
                raise ValueError("Imported caption model/template does not match this session")
        self.caption, self.caption_path = self._stage("pathgen", signature, s.pathgen_python, script,
            ["--image", self.image, "--model", s.pathgen_model], force, imported)
        self.explanation = self.explanation_path = None
        return self.caption

    def medgemma(self, force=False):
        if not self.caption_path:
            raise RuntimeError("Run Stage 1 first; it will reuse its saved caption.")
        s = self.settings
        local_base = Path("/projects/bfqi/jkim75/models/medgemma-4b-it")
        if s.medgemma_base == "google/medgemma-4b-it" and (local_base / "config.json").is_file():
            # A user may download missing weights after Stage 1 without restarting it.
            s.medgemma_base = str(local_base)
        base = Path(s.medgemma_base)
        signature = {"caption_sha256": _hash(self.caption_path),
            "base": _stamp(base) if base.is_dir() else s.medgemma_base,
            "adapter": _stamp(s.connect.parent / "medgem/medgemma_simplifier_adapter"),
            "instruction": _hash(s.connect.parent / "medgem/fewshot_instruction.txt"),
            "python": _stamp(s.medgemma_python)}
        # Preflight only when this stage actually needs a model job (handled below).
        self.explanation, self.explanation_path = self._stage("medgemma", signature,
            s.medgemma_python, s.connect / "simplify_medgemma.py",
            ["--caption-json", self.caption_path, "--base-model", s.medgemma_base], force)
        return self.explanation

    def musk(self, force=False):
        if not self.explanation_path:
            raise RuntimeError("Run Stage 2 first; it will reuse its saved explanation.")
        s = self.settings
        python = s.medgemma_python
        self.prediction, self.prediction_path = self._stage("musk",
            {"checkpoint": _stamp(s.checkpoint), "python": _stamp(python),
             "model_code": _hash(s.connect.parent / "musk/morphology_head.py")},
            python, s.connect.parent / "musk/infer_morphology.py",
            ["--image", self.image, "--checkpoint", s.checkpoint], force)
        self._unchanged()
        text, labels = render_report(self.explanation, self.prediction)
        record = dict(self.explanation, morphology=labels, musk_prediction=self.prediction,
                      musk_checkpoint=str(s.checkpoint), report=text)
        report_path = self.explanation_path.parent / ("report_" + _hash(self.prediction_path)[:16] + ".json")
        _write(report_path, record)
        record["report_path"] = str(report_path)
        return record
