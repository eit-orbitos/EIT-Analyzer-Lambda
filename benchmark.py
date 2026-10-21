"""Reproducible benchmark: neural vs. classical baselines over noise levels. Writes BENCHMARK_V4_2_1.json.
Protocol unchanged from V4.1 (same seeds, levels, methods and metric)."""
import json, os, platform, sys
import numpy as np, sklearn
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import eit_core as c

model, info = c.train_model()
out = {"app_version": c.APP_VERSION, "model": c.model_info_dict(info),
       "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
       "protocol": "200 synthetic signals per noise level, test seeds 50000-50199 via SeedSequence; metric = % MSE reduction vs noisy input",
       "results": {}}
for nz in (0.05, 0.1, 0.2, 0.35, 0.5, 0.8):
    acc = {}
    for s in range(50_000, 50_200):
        sr, nr = c.seeds_for(s)
        m = c.run_experiment(model, c.clean_signal(sr), nz, nr)["metrics"]
        for k in ("neural", "moving_average_5", "fft_lowpass_10"):
            acc.setdefault(k, []).append(m[f"reduction_pct_{k}"])
        # benchmark-only extra baseline: wider moving average (not shown in UI)
        sr2, nr2 = c.seeds_for(s); o = c.clean_signal(sr2); y = o + nr2.normal(0, nz, c.N)
        acc.setdefault("moving_average_9", []).append(100 * (c.mse(o, y) - c.mse(o, c.moving_average(y, 9))) / c.mse(o, y))
    out["results"][str(nz)] = {k: {"mean_reduction_pct": round(float(np.mean(v)), 2),
                                   "pct_runs_worse_than_noisy": round(100 * float(np.mean(np.array(v) < 0)), 1)}
                               for k, v in acc.items()}
path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BENCHMARK_V4_2_1.json")
with open(path, "w") as f:
    json.dump(out, f, indent=1, sort_keys=True)
print(json.dumps(out["results"], indent=1))
