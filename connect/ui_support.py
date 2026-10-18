"""UI helpers; reuse the notebook's runner without executing notebook cells."""
import ast
import hashlib
import json
from pathlib import Path

from integrated_report import FEATURE_NAMES


def notebook_defaults(connect):
    values = {"IMAGE_FILENAME": "TUM-YHTSLLSL.tif", "EXISTING_CAPTION": None}
    notebook = json.loads((Path(connect) / "Run_Pathology_Report.ipynb").read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id in values:
                        values[target.id] = node.value.value
    return values


def save_upload(connect, name, data):
    suffix = Path(name).suffix.lower()
    if suffix not in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}:
        raise ValueError("지원 형식: TIFF, PNG, JPEG")
    folder = Path(connect) / "reports" / "ui-inputs"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / (hashlib.sha256(data).hexdigest() + suffix)
    if not path.exists():
        path.write_bytes(data)
    return path


def prediction_rows(prediction):
    names = ["샘 구조 이상", "핵 이형성", "핵 배열 이상", "기질 침윤", "샘 내강 괴사"]
    words = {-1: "관찰되지 않음", 0: "불확실", 1: "관찰됨"}
    return [{"분석 항목": name, "항목 원문": FEATURE_NAMES[key],
             "모델 출력": words[int(prediction[key])],
             "선택 클래스 확률": float(prediction[key + "_confidence"])}
            for key, name in zip(FEATURE_NAMES, names)]


def run_stages(session, through, existing_caption=None, on_stage=None):
    """Restore preceding stages from disk on every retry, like the notebook."""
    result = None
    for stage in ("pathgen", "medgemma", "musk"):
        if on_stage:
            on_stage(stage)
        if stage == "pathgen":
            result = session.pathgen(existing_caption=existing_caption)
        else:
            result = getattr(session, stage)()
        if stage == through:
            return result
    raise ValueError("Unknown stage")
