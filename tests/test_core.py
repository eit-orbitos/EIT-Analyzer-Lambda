import os, sys, unittest
import json
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import eit_core as c

MODEL, INFO = c.train_model()
ONES = " ".join(["1"] * c.N).encode()


class TestParser(unittest.TestCase):
    def ok(self, data):
        v, n = c.parse_upload(data); return v, n
    def rej(self, data):
        with self.assertRaises(c.UploadError): c.parse_upload(data)
    def test_valid_space(self): self.assertEqual(self.ok(ONES)[0].size, c.N)
    def test_valid_newline_bom(self): self.assertEqual(self.ok(b"\xef\xbb\xbf" + b"\n".join([b"0.5"] * c.N))[0].size, c.N)
    def test_valid_sci(self): self.assertEqual(self.ok(" ".join(["-1.5e-3"] * c.N).encode())[0][0], -1.5e-3)
    def test_header(self): self.rej(b"signal\n" + ONES)
    def test_trailing_garbage(self): self.rej(ONES + b" abc")
    def test_nan(self): self.rej(" ".join(["1"] * 119 + ["nan"]).encode())
    def test_inf(self): self.rej(" ".join(["1"] * 119 + ["inf"]).encode())
    def test_hex(self): self.rej(" ".join(["0x10"] * c.N).encode())
    def test_count_low(self): self.rej(" ".join(["1"] * 119).encode())
    def test_count_high(self): self.rej(" ".join(["1"] * 121).encode())
    def test_not_utf8(self): self.rej(b"\xff\xfe" + ONES)
    def test_too_large(self): self.rej(b" " * (c.MAX_UPLOAD_BYTES + 1))
    def test_empty(self): self.rej(b"")
    def test_overflow_literal(self): self.rej(" ".join(["1e999"] * c.N).encode())
    def test_decimal_comma_warns(self):
        _, notes = self.ok(" ".join(["1,5"] * 60).encode())
        self.assertTrue(any("decimal comma" in n for n in notes))
    def test_constant_warns(self): self.assertTrue(any("constant" in n for n in self.ok(ONES)[1]))


class TestModel(unittest.TestCase):
    def test_converged(self): self.assertTrue(INFO.converged)
    def test_deterministic(self):
        m2, _ = c.train_model(); y = np.random.default_rng(1).normal(size=c.N)
        np.testing.assert_array_equal(c.neural_reconstruct(MODEL, y), c.neural_reconstruct(m2, y))
    def test_shape_finite(self):
        r = c.neural_reconstruct(MODEL, np.zeros(c.N)); self.assertEqual(r.shape, (c.N,)); self.assertTrue(np.all(np.isfinite(r)))
    def test_offset_equivariance(self):
        y = np.random.default_rng(3).normal(size=c.N)
        np.testing.assert_allclose(c.neural_reconstruct(MODEL, y + 5) - 5, c.neural_reconstruct(MODEL, y), atol=1e-9)
    def test_seed_streams_distinct(self):
        a, b = c.seeds_for(0); x, y = c.seeds_for(100000)
        self.assertFalse(np.array_equal(a.random(5), y.random(5)))
    def test_experiment_reproducible(self):
        def once():
            s, n = c.seeds_for(999); return c.run_experiment(MODEL, c.clean_signal(s), 0.35, n)["metrics"]
        self.assertEqual(once(), once())


class TestHonestBenchmark(unittest.TestCase):
    """Guards the claims in README: at noise>=0.35 the neural model beats noisy input in every case and
    beats moving-average-5 on average; at noise 0.05 it is usually WORSE than noisy input (documented)."""
    def stats(self, nz, n=60):
        nn, ma = [], []
        for s in range(n):
            sr, nr = c.seeds_for(10_000 + s)
            m = c.run_experiment(MODEL, c.clean_signal(sr), nz, nr)["metrics"]
            nn.append(m["reduction_pct_neural"]); ma.append(m["reduction_pct_moving_average_5"])
        return np.array(nn), np.array(ma)
    def test_mid_noise(self):
        nn, ma = self.stats(0.35); self.assertTrue(np.all(nn > 0)); self.assertGreater(nn.mean(), ma.mean())
    def test_low_noise_limitation_documented(self):
        nn, _ = self.stats(0.05); self.assertGreater(np.mean(nn < 0), 0.5)


def _vals(token, n=None):
    return " ".join([token] * (n or c.N)).encode()


class TestKI03Separators(unittest.TestCase):
    def test_semicolon_decimal_warns(self):
        v, notes = c.parse_upload(_vals("1;5", 60))
        self.assertEqual(v.size, c.N); self.assertTrue(any("';'" in n and "decimal" in n for n in notes))
    def test_mixed_comma_semicolon_warns(self):
        _, notes = c.parse_upload(_vals("1,5;2,3", 30))
        self.assertTrue(any("','" in n and "';'" in n for n in notes))
    def test_semicolon_with_dot_decimals_no_warning(self):
        _, notes = c.parse_upload(";".join(["0.5", "1.5"] * 60).encode())
        self.assertFalse(any("decimal" in n for n in notes))
    def test_comma_warning_still_present(self):
        _, notes = c.parse_upload(_vals("1,5", 60))
        self.assertTrue(any("decimal comma" in n for n in notes))


class TestKI04Bounds(unittest.TestCase):
    def test_upload_at_bound_accepted(self):
        v, _ = c.parse_upload(" ".join(["1e100", "-1e100"] * 60).encode()); self.assertEqual(float(np.max(np.abs(v))), 1e100)
    def test_upload_above_bound_rejected(self):
        for t in ("1.0001e100", "-1e101", "1e200", "1e308"):
            with self.assertRaises(c.UploadError, msg=t): c.parse_upload(_vals(t))
    def test_run_rejects_out_of_domain(self):
        _, nr = c.seeds_for(1)
        for bad in (np.full(c.N, 1e101), np.full(c.N, np.inf), np.full(c.N, np.nan), np.zeros(c.N - 1)):
            with self.assertRaises(ValueError): c.run_experiment(MODEL, bad, 0.35, nr)
        for nz in (float("nan"), float("inf"), -0.1):
            with self.assertRaises(ValueError): c.run_experiment(MODEL, np.zeros(c.N), nz, nr)
    def test_max_bound_signal_metrics_finite_and_strict_json(self):
        sr, _ = c.seeds_for(1); s = c.clean_signal(sr); s = s / np.max(np.abs(s)) * 1e100
        res = c.run_experiment(MODEL, s, 0.35, c.seeds_for(5)[1])
        self.assertEqual(res["metrics"]["non_finite_metrics"], [])
        json.loads(c.to_strict_json(res["metrics"]))
    def test_non_finite_guard(self):
        """Huge noise overflows MSE: metrics become None, are listed, and export stays strict JSON."""
        res = c.run_experiment(MODEL, np.zeros(c.N), 1e200, c.seeds_for(5)[1])
        m = res["metrics"]
        self.assertIn("mse_noisy", m["non_finite_metrics"]); self.assertIsNone(m["mse_noisy"])
        self.assertIsNone(m["best_method"]); self.assertFalse(m["neural_beats_all_baselines"])
        self.assertTrue(res["notes"])
        s = c.to_strict_json(m); self.assertNotIn("Infinity", s); self.assertNotIn("NaN", s); json.loads(s)
    def test_strict_json_refuses_non_finite(self):
        for bad in ({"x": float("inf")}, {"x": float("nan")}, [np.float64("-inf")]):
            with self.assertRaises(c.ExportError): c.to_strict_json(bad)
    def test_noise_absorption_reported(self):
        res = c.run_experiment(MODEL, np.full(c.N, 1e100), 0.35, c.seeds_for(5)[1])
        self.assertEqual(res["metrics"]["mse_noisy"], 0.0)
        self.assertTrue(any("absorbed" in n for n in res["notes"]))


class TestKI05Underflow(unittest.TestCase):
    def test_nonzero_underflow_rejected(self):
        for t in ("1e-999", "-1e-400", "0.0001e-330", "5E-325"):
            with self.assertRaises(c.UploadError, msg=t): c.parse_upload(_vals(t))
    def test_true_zero_literals_accepted(self):
        for t in ("0", "0.000", "0e-999", "-0.0E5"):
            v, _ = c.parse_upload(_vals(t)); self.assertTrue(np.all(v == 0.0), t)
    def test_subnormal_nonzero_accepted(self):
        v, _ = c.parse_upload(_vals("1e-320")); self.assertGreater(v[0], 0.0)


class TestKI06Display(unittest.TestCase):
    def test_format_normal(self):
        self.assertEqual(c.format_pct(81.38), "81.4%"); self.assertEqual(c.format_pct(-129.97), "-130.0%")
    def test_format_extreme_scientific(self):
        s = c.format_pct(-6174001887736.2); self.assertIn("e+", s); self.assertTrue(s.endswith("%"))
        self.assertEqual(c.format_pct(None), "n/a")
    def test_stored_metric_not_clamped(self):
        sr, _ = c.seeds_for(1); s = c.clean_signal(sr) * 1e6
        m = c.run_experiment(MODEL, s, 0.35, c.seeds_for(777)[1])["metrics"]
        self.assertTrue(c.is_extreme_pct(m["reduction_pct_neural"])); self.assertLess(m["reduction_pct_neural"], -1e6)


if __name__ == "__main__":
    unittest.main()
