"""Quiet Slurm wrapper for the one-cell pathology report notebook (no ML imports)."""
import hashlib
import json
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


def find_connect(start=None):
    start = Path(start or Path.cwd()).resolve()
    for parent in (start, *start.parents):
        for candidate in (parent, parent / "connect"):
            if (candidate / "report_delta.sbatch").is_file() and (candidate / "integrated_report.py").is_file():
                return candidate
    raise FileNotFoundError("Open the repository connect folder in VS Code and run the notebook.")


@dataclass
class Settings:
    connect: Path
    image_root: Path = Path("/projects/bfqi/shared/datasets/NCT-CRC-HE-100K/data-samples/TUM")
    checkpoint: Path | None = None
    pathgen_model: Path = Path("/projects/bfqi/jkim75/models/PathGen-LLaVA")
    pathgen_python: Path | None = None
    medgemma_python: Path = Path("/sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/bin/python")
    poll_seconds: float = 10
    timeout_seconds: float = 4 * 60 * 60

    def __post_init__(self):
        self.connect = Path(self.connect).resolve()
        self.checkpoint = Path(self.checkpoint or self.connect.parent / "musk/runs/head_slurm_22984898/best.pt")
        self.pathgen_python = Path(self.pathgen_python or self.connect / ".venv-pathgen/bin/python")


def _command(args, cwd):
    result = subprocess.run(list(map(str, args)), cwd=cwd, text=True,
                            capture_output=True, timeout=30)
    return result


def _diagnostic(job, logs):
    messages = []
    for suffix in ("err", "out"):
        path = logs / f"notebook-{job}.{suffix}"
        if path.exists():
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
            if lines:
                messages.append(f"{path}\n" + "\n".join(line[-1500:] for line in lines[-20:]))
    return "\n\n".join(messages) or f"Log directory: {logs}"


def run_report(filename, settings=None):
    """Submit one GPU job and return report.json; normal execution prints no logs.

    Runs on a Delta CPU/login notebook kernel; GPU inference happens only in Slurm.
    Interrupting the cell requests cancellation of this wrapper's own submitted job.
    """
    settings = settings or Settings(find_connect())
    filename = filename.strip()
    if not filename or filename in {".", ".."} or "/" in filename or "\\" in filename:
        raise ValueError("Enter an image filename without a directory, e.g. TUM-YHTSLLSL.tif")
    if Path(filename).suffix.lower() not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise ValueError("Supported image formats: tif, tiff, png, jpg, jpeg")
    if settings.poll_seconds <= 0 or settings.timeout_seconds <= 0:
        raise ValueError("Polling interval and timeout must be positive")
    image = (settings.image_root / filename).resolve(strict=True)
    required = [settings.checkpoint, settings.pathgen_python, settings.medgemma_python,
                settings.pathgen_model / "config.json", settings.connect / "report_delta.sbatch",
                settings.connect.parent / "musk/infer_morphology.py",
                settings.connect.parent / "medgem/medgemma_simplifier_adapter/adapter_config.json"]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(f"Required file not found: {path}")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    logs = settings.connect / "logs"
    logs.mkdir(exist_ok=True)
    # The pipeline requires a new directory; do not create this output directory here.
    output = settings.connect / "reports" / f"notebook_{uuid.uuid4().hex}"
    command = ["sbatch", "--parsable", "--export=ALL",
        "--output", logs / "notebook-%j.out", "--error", logs / "notebook-%j.err",
        settings.connect / "report_delta.sbatch",
        "--image", image, "--musk-checkpoint", settings.checkpoint.resolve(),
        "--pathgen-model", settings.pathgen_model,
        "--pathgen-python", settings.pathgen_python,
        "--medgemma-python", settings.medgemma_python,
        "--output-dir", output]
    submitted = _command(command, settings.connect)
    if submitted.returncode:
        raise RuntimeError("Slurm submission failed:\n" + submitted.stderr.strip())
    job = submitted.stdout.strip().split(";")[0]
    if not re.fullmatch(r"\d+", job):
        raise RuntimeError(f"Could not determine the Slurm job ID: {submitted.stdout!r}")
    (logs / f"notebook-{job}.json").write_text(json.dumps({
        "job_id": job, "image": str(image), "output_dir": str(output)}, indent=2), encoding="utf-8")
    deadline = time.monotonic() + settings.timeout_seconds
    terminal = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY",
                "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}
    missing_since = None
    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Wait timed out. Job ID: {job}")
            accounting = _command(["sacct", "-n", "-X", "-P", "-j", job,
                                    "--format=JobIDRaw,State,ExitCode"], settings.connect)
            state = exit_code = None
            if accounting.returncode == 0:
                for line in accounting.stdout.splitlines():
                    fields = line.strip().split("|")
                    if len(fields) >= 3 and fields[0] == job:
                        state = fields[1].split()[0].rstrip("+")
                        exit_code = fields[2]
                        break
            if state in terminal:
                if state != "COMPLETED" or exit_code != "0:0":
                    raise RuntimeError(f"Job {job}: {state} ({exit_code})\n" + _diagnostic(job, logs))
                report_file = output / "report.json"
                if not report_file.is_file():
                    raise RuntimeError(f"No final report file found for job {job}.\n" + _diagnostic(job, logs))
                record = json.loads(report_file.read_text(encoding="utf-8"))
                if record.get("image_sha256") != digest or Path(record["image_path"]).resolve() != image:
                    raise RuntimeError("The report image does not match the input image.")
                if not isinstance(record.get("report"), str) or not record["report"].strip():
                    raise RuntimeError("The final report is empty.")
                record["slurm_job_id"] = job
                record["output_dir"] = str(output)
                return record
            # Accounting can lag behind queue disappearance. Never treat absence as success.
            queue = _command(["squeue", "-h", "-j", job, "-o", "%T"], settings.connect)
            if queue.returncode == 0 and not queue.stdout.strip() and state is None:
                missing_since = missing_since or time.monotonic()
                if time.monotonic() - missing_since > 120:
                    raise RuntimeError(f"Could not determine the final status of job {job}. Check sacct.\n" + _diagnostic(job, logs))
            else:
                missing_since = None
            time.sleep(settings.poll_seconds)
    except (KeyboardInterrupt, TimeoutError):
        cancelled = _command(["scancel", job], settings.connect)
        if cancelled.returncode:
            raise RuntimeError(f"Automatic cancellation failed for job {job}. Check using scancel {job}.")
        raise
