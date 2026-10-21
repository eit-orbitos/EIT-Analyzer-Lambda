"""KI-01 / HI-03: license bytes pinned and metadata free of TODO placeholders."""
import hashlib, json, os, unittest

ROOT = os.path.join(os.path.dirname(__file__), "..")
# PolyForm Noncommercial 1.0.0, copied verbatim from github.com/eit-orbitos/EIT-Program-Lang-
# commit 6ff36a614fc060b67bade549761e690bd21b551d (git blob SHA-1 verified by Tasklet 2026-10-20).
LICENSE_SHA256 = "c0ea4a896d2c8c394b29f9427589996db826cd501c512279ff0ed3ef48fabbe5"


def read(name, mode="r"):
    with open(os.path.join(ROOT, name), mode) as f:
        return f.read()


class TestReleaseMetadata(unittest.TestCase):
    def test_license_bytes_pinned(self):
        self.assertEqual(hashlib.sha256(read("LICENSE", "rb")).hexdigest(), LICENSE_SHA256)
    def test_license_scope_files_present(self):
        for f in ("LICENSE-DOCUMENTATION.md", "COMMERCIAL-USE-POLICY.md", "NOTICE"):
            self.assertIn("EIT Analyzer Lambda", read(f), f)
            self.assertNotIn("EIT Program Lang", read(f), f)
    def test_no_todo_placeholders(self):
        for f in ("CITATION.cff", ".zenodo.json", "README.md", "LICENSE-DOCUMENTATION.md", "NOTICE",
                  "COMMERCIAL-USE-POLICY.md"):
            self.assertNotIn("TODO", read(f), f)
            self.assertNotIn("AUTHOR_DECISION_REQUIRED", read(f), f)
    def test_license_identifiers_consistent(self):
        z = json.loads(read(".zenodo.json"))
        self.assertEqual(z["license"], "polyform-noncommercial-1.0.0")  # Zenodo vocabulary id
        self.assertIn("license: PolyForm-Noncommercial-1.0.0", read("CITATION.cff"))
        self.assertNotIn("doi", read("CITATION.cff").lower())
    def test_versions_consistent(self):
        import sys; sys.path.insert(0, ROOT); import eit_core
        self.assertIn(eit_core.APP_VERSION, read("CITATION.cff"))
        self.assertEqual(json.loads(read(".zenodo.json"))["version"], eit_core.APP_VERSION)
        self.assertIn(eit_core.APP_VERSION, read("CHANGELOG.md"))


class TestStrictJSON(unittest.TestCase):
    """V421-04: all JSON shipped in the package is strict RFC 8259 JSON."""
    def test_all_json_strict(self):
        def no_const(x): raise ValueError(f"non-finite constant {x}")
        def no_dup(pairs):
            keys = [k for k, _ in pairs]
            if len(keys) != len(set(keys)): raise ValueError("duplicate key")
            return dict(pairs)
        found = [f for f in os.listdir(ROOT) if f.endswith(".json")]
        self.assertGreaterEqual(len(found), 2)
        for f in found:
            json.loads(read(f), parse_constant=no_const, object_pairs_hook=no_dup)


if __name__ == "__main__":
    unittest.main()
