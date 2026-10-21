# Changelog

## 4.2.1 (release candidate; release status recorded outside the archive)
Delta against V4.2 (`EIT_Analyzer_V4.2_candidate.zip`, SHA-256 `b2b2d0a0…806179a`). No change to license terms,
license texts, benchmark protocol, parser rules or metrics.
- V421-01 `.zenodo.json` license id set to Zenodo vocabulary id `polyform-noncommercial-1.0.0` (Zenodo API, checked
  2026-10-21). `CITATION.cff` keeps the SPDX identifier `PolyForm-Noncommercial-1.0.0`.
- V421-02 Version `4.2.1` without `-candidate`; README no longer states review/freeze status inside the archive, so the
  reviewed bytes can be frozen unchanged.
- V421-03 Streamlit `use_container_width=True` replaced by `width="stretch"` (removes deprecation warnings; pinned 1.66.0).
- V421-04 New test: every JSON file in the package parses under a strict parser (no NaN/Infinity, no duplicate keys).
- Benchmark output renamed `BENCHMARK_V4_2_1.json`.

## 4.2.0-candidate (2026-10-20) — correction candidate, not frozen
Delta against V4.1 (`EIT_Analyzer_V4.1_candidate.zip`, SHA-256 `9f81f533…ada014`), per author decisions D1–D4:
- KI-01 / HI-03 Added `LICENSE` (PolyForm Noncommercial 1.0.0, verbatim from pinned commit `6ff36a6`),
  Analyzer-specific `LICENSE-DOCUMENTATION.md`, `COMMERCIAL-USE-POLICY.md`, `NOTICE`; README, `CITATION.cff`,
  `.zenodo.json` synchronized (no DOI, no release date). Test pins the LICENSE SHA-256.
- KI-03 Decimal-separator warning now also covers `;` (and mixed `,`/`;`).
- KI-04 Documented bound |x| ≤ 1e100 for uploads and experiment inputs; `noise_std` must be finite and ≥ 0;
  non-finite metrics become `null` and are listed in `non_finite_metrics`; all JSON export is strict
  (`allow_nan=False`); float64 noise absorption is reported.
- KI-05 Non-zero literals that underflow to 0.0 are rejected.
- KI-06 Reductions above ±1000% shown in scientific notation with a warning (display only; stored values unchanged).
- App file renamed `eit_app_v4_1.py` → `eit_app_v4_2.py`; export file names use `v4_2`.
- Benchmark protocol unchanged; output file renamed `BENCHMARK_V4_2.json`.

## 4.1.0-candidate (2026-10-20) — correction candidate, not frozen
Fixes against V4 (`eit_app_v4.py`, SHA-256 `ff7b36bb…49bb19`):
- EA-01 Classical baselines (moving average 5, FFT low-pass 10) shown next to the neural result; warning when a baseline wins.
- EA-02 Model trained over the full noise range 0.05–0.8 (V4: only 0.35) with 1500 samples; converges (V4: ConvergenceWarning at 220 iterations).
- EA-03 Per-signal z-score scaling: offset/scale shifts no longer break the model (V4: +2 offset → worse in 100% of runs).
- EA-04 Strict upload parser: rejects headers, nan/inf/hex, wrong count, non-UTF-8, files > 64 KiB; warns on probable decimal comma (V4 silently read `1,5` as two values 1 and 5).
- EA-05 Independent signal/noise random streams via SeedSequence (V4: noise stream of seed s = signal stream of seed s+100000).
- EA-06 Experiment JSON records app version, model settings, convergence and library versions.
- EA-07 Pinned dependency versions; `__pycache__` removed from the package.
- EA-08 Core logic split into `eit_core.py`; 25 unit/headless tests; reproducible `benchmark.py`.
- EA-09 README states measured limitations, including that a 9-point moving average beats the neural model.
