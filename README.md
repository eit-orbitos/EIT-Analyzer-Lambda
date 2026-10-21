# EIT Analyzer Lambda V4.2.1

Research and teaching prototype. It adds synthetic noise to a signal, then compares a small neural network
(scikit-learn `MLPRegressor`) with simple classical filters. It includes an interactive plot, a decorative 3D view,
session memory and CSV/JSON export.

Release status (candidate, frozen or published) is not stated inside this archive. It is recorded by the author in a
separate release receipt that names the exact SHA-256 of the archive. Earlier versions (V4, V4.1, V4.2) are preserved
unchanged.

## Install and run
```bash
python -m pip install -r requirements.txt
python -m streamlit run eit_app_v4_2.py
```
Open http://localhost:8501. Tested with Python 3.12.15 on Linux (versions pinned in `requirements.txt`).

## Tests and benchmark
```bash
python -m unittest discover -s tests -t .      # full suite incl. Streamlit headless; also under -O / -OO
python benchmark.py                             # writes BENCHMARK_V4_2_1.json (protocol unchanged since V4.1)
```

## Upload format
Exactly 120 numbers separated by newlines, spaces, `;` or `,`. Use `.` as the decimal separator. No header,
no `nan`/`inf`/hex, max 64 KiB, UTF-8. Every value must satisfy |x| ≤ 1e100 (a software bound that keeps all
error metrics finite in float64, not a physical limit). A non-zero value that is too small to represent
(e.g. `1e-999`) is rejected instead of silently becoming 0. If all values are integers and `,` or `;` appear,
the app warns that a decimal separator may have been misread.

Uploaded values are treated as the clean reference and synthetic noise is added on top, so for real noisy
measurements the error figures do not describe the true error. At very large magnitudes the added noise can be
absorbed by float64 rounding; the app then says so.

## Export
JSON exports are strict RFC 8259 JSON: `NaN`/`Infinity` are never written. A metric that cannot be represented
is exported as `null` and listed in `non_finite_metrics`. Reductions above ±1000% are shown in scientific notation
with a warning; the exported values are not clamped.

## What the benchmark shows (200 synthetic signals per noise level)
| Noise std | Neural | Moving avg 5 | Moving avg 9 | FFT low-pass 10 |
|---|---|---|---|---|
| 0.05 | −130% (worse in 88% of runs) | +59% | −96% | −137% |
| 0.10 | +32% | +74% | +42% | +29% |
| 0.20 | +72% | +78% | +76% | +70% |
| 0.35 | +81% | +79% | +84% | +80% |
| 0.50 | +84% | +79% | +86% | +82% |
| 0.80 | +85% | +79% | +87% | +83% |

A plain 9-point moving average beats the neural model at every noise level from 0.20 up. At low noise the neural
model makes the signal worse. The neural component therefore shows **no demonstrated advantage** over simple filters
on this data. It also fails on signals far from its training family (steps, large amplitudes relative to noise).

## Research boundary
- The 3D view is decorative. It is not a hologram and not a measured energy field.
- Λ is a label, not a measured constant, and is not used in any calculation.
- This software tests no EIT equation and is not evidence for or against EIT theory, time travel or consciousness transfer.
- Session memory is lost when the server restarts.

## License and citation
- Code: **PolyForm Noncommercial License 1.0.0** — `LICENSE` (verbatim text).
- Documentation: **CC BY-NC-SA 4.0** — scope in `LICENSE-DOCUMENTATION.md`.
- Commercial use requires a separate written agreement — `COMMERCIAL-USE-POLICY.md`, `NOTICE`.
- Citation metadata: `CITATION.cff`. No DOI has been assigned.

Copyright (c) 2026 Mr. Toni Mladenovski (EITNetworks LLC).
