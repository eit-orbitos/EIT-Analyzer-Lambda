"""EIT Analyzer Lambda V4.2.1 — pure computational core (no UI).

Research prototype. Synthetic signals, neural denoising, classical baselines.
Nothing here measures a physical field or tests an EIT physical law.
"""
from __future__ import annotations

import json
import math
import re
import warnings
from dataclasses import dataclass, asdict

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor

APP_VERSION = "4.2.1"
N = 120
TRAIN_SEED = 42
TRAIN_SAMPLES = 1500
TRAIN_NOISE_RANGE = (0.05, 0.8)
MAX_UPLOAD_BYTES = 64 * 1024
# KI-04: documented software bound on signal magnitude. Keeps every MSE finite in float64
# (|error| <= ~2e100 -> squared <= ~4e200 < 1.8e308). Not a physical limit.
MAX_ABS_VALUE = 1e100
# KI-06: display-only threshold; stored metrics are never clamped or altered.
EXTREME_PCT_DISPLAY = 1000.0
_FLOAT_RE = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")
_INT_RE = re.compile(r"[+-]?\d+")


@dataclass(frozen=True)
class ModelInfo:
    hidden_layer_sizes: tuple
    train_seed: int
    train_samples: int
    train_noise_range: tuple
    n_iter: int
    converged: bool
    input_scaling: str = "per-signal z-score (mean/std of noisy input), inverted after prediction"


def clean_signal(rng: np.random.Generator, n: int = N) -> np.ndarray:
    x = np.linspace(0, 4 * np.pi, n)
    return (rng.uniform(.6, 1.1) * np.sin(rng.uniform(.7, 1.3) * x + rng.uniform(0, 6.28))
            + rng.uniform(.15, .5) * np.sin(rng.uniform(2, 3.5) * x + rng.uniform(0, 6.28)))


def _zscore(y: np.ndarray):
    mu = float(np.mean(y)); sd = float(np.std(y))
    if not np.isfinite(sd) or sd < 1e-12:
        sd = 1.0
    return (y - mu) / sd, mu, sd


def train_model():
    """Deterministic training over the full noise range, on z-scored inputs/targets."""
    rng = np.random.default_rng(TRAIN_SEED)
    targets = np.stack([clean_signal(rng) for _ in range(TRAIN_SAMPLES)])
    sig = rng.uniform(*TRAIN_NOISE_RANGE, size=(TRAIN_SAMPLES, 1))
    noisy = targets + rng.normal(0, 1, targets.shape) * sig
    X, Y = [], []
    for t, y in zip(targets, noisy):
        zy, mu, sd = _zscore(y)
        X.append(zy); Y.append((t - mu) / sd)
    model = MLPRegressor(hidden_layer_sizes=(128,), max_iter=800, random_state=TRAIN_SEED,
                         early_stopping=True, n_iter_no_change=20)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(np.array(X), np.array(Y))
    converged = not any(issubclass(x.category, ConvergenceWarning) for x in w)
    info = ModelInfo((128,), TRAIN_SEED, TRAIN_SAMPLES, TRAIN_NOISE_RANGE, int(model.n_iter_), converged)
    return model, info


def neural_reconstruct(model, y: np.ndarray) -> np.ndarray:
    zy, mu, sd = _zscore(y)
    return model.predict(zy.reshape(1, -1))[0] * sd + mu


def moving_average(y: np.ndarray, k: int = 5) -> np.ndarray:
    p = np.pad(y, k // 2, mode="edge")
    return np.convolve(p, np.ones(k) / k, mode="valid")


def fft_lowpass(y: np.ndarray, keep: int = 10) -> np.ndarray:
    F = np.fft.rfft(y); F[keep:] = 0
    return np.fft.irfft(F, n=len(y))


def seeds_for(test_seed: int):
    """Independent signal/noise streams; no cross-seed stream reuse (fixes V4 seed+100000 overlap)."""
    ss = np.random.SeedSequence([20261020, int(test_seed)])
    s_sig, s_noise = ss.spawn(2)
    return np.random.default_rng(s_sig), np.random.default_rng(s_noise)


def mse(a, b) -> float:
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def _finite_or_none(x: float):
    return x if math.isfinite(x) else None


def run_experiment(model, original: np.ndarray, noise_std: float, noise_rng: np.random.Generator) -> dict:
    """Run one experiment. Raises ValueError for inputs outside the documented domain (KI-04).

    Any metric that is still non-finite is stored as None and listed in metrics['non_finite_metrics'];
    such a metric never takes part in choosing best_method.
    """
    original = np.asarray(original, dtype=float)
    if original.shape != (N,):
        raise ValueError(f"original must have shape ({N},)")
    if not np.all(np.isfinite(original)) or float(np.max(np.abs(original))) > MAX_ABS_VALUE:
        raise ValueError(f"original values must be finite with |x| <= {MAX_ABS_VALUE:g}")
    noise_std = float(noise_std)
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std must be finite and >= 0")
    noisy = original + noise_rng.normal(0, noise_std, original.shape[0])
    # Overflow can only occur for noise_std far above the documented range; it is caught below as non-finite.
    with np.errstate(over="ignore", invalid="ignore"):
        recon = {
            "neural": neural_reconstruct(model, noisy),
            "moving_average_5": moving_average(noisy, 5),
            "fft_lowpass_10": fft_lowpass(noisy, 10),
        }
        base = mse(original, noisy)
        raw = {k: mse(original, v) for k, v in recon.items()}
    non_finite: list[str] = []
    metrics = {"mse_noisy": _finite_or_none(base)}
    if metrics["mse_noisy"] is None:
        non_finite.append("mse_noisy")
    for k, v in raw.items():
        metrics[f"mse_{k}"] = _finite_or_none(v)
        if metrics[f"mse_{k}"] is None:
            non_finite.append(f"mse_{k}")
        if metrics["mse_noisy"] is None or metrics[f"mse_{k}"] is None:
            red = None
        else:
            red = 100 * (base - v) / base if base > 0 else 0.0
            red = _finite_or_none(red)
        metrics[f"reduction_pct_{k}"] = red
        if red is None:
            non_finite.append(f"reduction_pct_{k}")
    finite_methods = [k for k in recon if metrics[f"mse_{k}"] is not None]
    best = min(finite_methods, key=lambda k: metrics[f"mse_{k}"]) if finite_methods else None
    metrics["best_method"] = best
    metrics["neural_beats_all_baselines"] = best == "neural"
    metrics["non_finite_metrics"] = non_finite
    notes: list[str] = []
    if non_finite:
        notes.append("Some metrics overflowed float64 and are reported as null: " + ", ".join(non_finite) + ".")
    if noise_std > 0 and base == 0:
        notes.append("The added noise was completely absorbed by float64 rounding at this signal magnitude; "
                     "the experiment effectively ran without noise and reductions are reported as 0.")
    return {"noisy": noisy, "recon": recon, "metrics": metrics, "notes": notes}


class UploadError(ValueError):
    pass


class ExportError(ValueError):
    pass


def parse_upload(data: bytes) -> tuple[np.ndarray, list[str]]:
    """Strict parser. Returns (values, warnings). Raises UploadError on any invalid input.

    Accepted separators: newline, whitespace, ';' and ','. Decimal separator must be '.'.
    Every token must be a plain decimal number; no headers, hex, nan, inf.
    Values must satisfy |x| <= MAX_ABS_VALUE (KI-04). A non-zero literal that rounds to 0.0
    (float64 underflow, e.g. 1e-999) is rejected (KI-05).
    """
    notes: list[str] = []
    if len(data) > MAX_UPLOAD_BYTES:
        raise UploadError(f"file larger than {MAX_UPLOAD_BYTES} bytes")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        raise UploadError("file is not valid UTF-8 text")
    tokens = [t for t in re.split(r"[\s;,]+", text.strip()) if t]
    bad = [t for t in tokens if not _FLOAT_RE.match(t)]
    if bad:
        raise UploadError(f"non-numeric token(s), e.g. {bad[0][:20]!r}; headers, nan, inf and hex are not allowed")
    if len(tokens) != N:
        raise UploadError(f"expected exactly {N} values, found {len(tokens)}")
    vals = np.array([float(t) for t in tokens])
    if not np.all(np.isfinite(vals)):
        raise UploadError("values must be finite")
    if float(np.max(np.abs(vals))) > MAX_ABS_VALUE:
        raise UploadError(f"values must satisfy |x| <= {MAX_ABS_VALUE:g} (software bound, not a physical limit)")
    for t, v in zip(tokens, vals):
        if v == 0.0 and re.search(r"[1-9]", t.split("e")[0].split("E")[0]):
            raise UploadError(f"value {t[:20]!r} is non-zero but underflows to 0.0 in float64")
    seps = [s for s in (",", ";") if s in text]
    if seps and all(_INT_RE.fullmatch(t) for t in tokens):
        shown = " and ".join(f"'{s}'" for s in seps)
        notes.append(f"All values are integers and {shown} separators are present. If your file uses a decimal "
                     "comma (e.g. 1,5) or another decimal separator, it has been misread: use '.' for decimals.")
    if float(np.std(vals)) < 1e-12:
        notes.append("Signal is constant; reduction percentages are not meaningful.")
    return vals, notes


def model_info_dict(info: ModelInfo) -> dict:
    d = asdict(info); d["hidden_layer_sizes"] = list(d["hidden_layer_sizes"])
    d["train_noise_range"] = list(d["train_noise_range"]); return d


def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def to_strict_json(obj, **kw) -> str:
    """Strict RFC 8259 JSON: NaN/Infinity are never emitted (KI-04). Raises ExportError instead."""
    try:
        return json.dumps(obj, allow_nan=False, default=_plain, **kw)
    except ValueError as exc:
        raise ExportError(f"export refused: {exc}") from exc


def format_pct(p) -> str:
    """Display-only formatting (KI-06). The stored metric value is not changed."""
    if p is None:
        return "n/a"
    if abs(p) > EXTREME_PCT_DISPLAY:
        return f"{p:.2e}%"
    return f"{p:.1f}%"


def is_extreme_pct(p) -> bool:
    return p is not None and abs(p) > EXTREME_PCT_DISPLAY
