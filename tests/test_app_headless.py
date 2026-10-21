"""Headless Streamlit run (no server, no network)."""
import os, unittest
from streamlit.testing.v1 import AppTest
APP = os.path.join(os.path.dirname(__file__), "..", "eit_app_v4_2.py")


class TestApp(unittest.TestCase):
    def test_default_and_controls(self):
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertEqual(len(at.exception), 0)
        self.assertEqual(len(at.metric), 4)
        for nz in (0.05, 0.8):
            at.sidebar.slider[0].set_value(nz).run(); self.assertEqual(len(at.exception), 0)
        at.sidebar.number_input[0].set_value(100000).run(); self.assertEqual(len(at.exception), 0)
        at.button[0].click().run(); self.assertEqual(len(at.exception), 0)
        at.sidebar.radio[0].set_value("Upload CSV/TXT").run()
        self.assertEqual(len(at.exception), 0); self.assertTrue(any("Upload" in i.value for i in at.info))

    def test_metric_display_format(self):
        """KI-06: reductions are rendered through core.format_pct (always end with '%')."""
        at = AppTest.from_file(APP, default_timeout=120).run()
        self.assertEqual(len(at.exception), 0)
        for mt in at.metric[1:]:
            self.assertTrue(str(mt.value).endswith("%"), mt.value)

    def test_app_exports_only_strict_json(self):
        """KI-04: the app must not call json.dumps directly; all JSON goes through core.to_strict_json."""
        with open(APP, encoding="utf-8") as f:
            src = f.read()
        self.assertNotIn("json.dumps", src)
        self.assertGreaterEqual(src.count("core.to_strict_json("), 2)


if __name__ == "__main__":
    unittest.main()
