import tempfile
import unittest
from pathlib import Path

from ui_support import notebook_defaults, run_stages, save_upload


class FakeSession:
    def __init__(self, fail=False):
        self.calls = []
        self.fail = fail

    def pathgen(self, existing_caption=None):
        self.calls.append(("pathgen", existing_caption))
        return "caption"

    def medgemma(self):
        self.calls.append(("medgemma", None))
        if self.fail:
            raise RuntimeError("GPU stage failed")
        return "explanation"

    def musk(self):
        self.calls.append(("musk", None))
        return "report"


class WrapperTests(unittest.TestCase):
    def test_notebook_defaults(self):
        defaults = notebook_defaults(Path(__file__).parent)
        self.assertEqual(defaults["IMAGE_FILENAME"], "TUM-YHTSLLSL.tif")
        self.assertTrue(defaults["EXISTING_CAPTION"].endswith("caption.json"))

    def test_stage_order_and_caption_import(self):
        session = FakeSession()
        stages = []
        self.assertEqual(run_stages(session, "musk", "old.json", stages.append), "report")
        self.assertEqual(stages, ["pathgen", "medgemma", "musk"])
        self.assertEqual(session.calls[0], ("pathgen", "old.json"))

    def test_partial_stops_before_musk(self):
        session = FakeSession()
        self.assertEqual(run_stages(session, "medgemma"), "explanation")
        self.assertEqual(len(session.calls), 2)

    def test_failure_does_not_run_downstream(self):
        session = FakeSession(fail=True)
        with self.assertRaises(RuntimeError):
            run_stages(session, "musk")
        self.assertEqual(len(session.calls), 2)

    def test_upload_name_cannot_escape_storage(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as directory:
            root = Path(directory)
            first = save_upload(root, "../../image.png", b"example")
            second = save_upload(root, "other.png", b"example")
            self.assertEqual(first, second)
            self.assertEqual(first.parent, root / "reports/ui-inputs")
            self.assertEqual(first.read_bytes(), b"example")
            with self.assertRaises(ValueError):
                save_upload(root, "image.exe", b"example")


if __name__ == "__main__":
    unittest.main()
