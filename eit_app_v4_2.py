"""EIT Analyzer Lambda V4.2.1 — Streamlit UI. Research prototype; see README for boundaries."""
import platform
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sklearn
import streamlit as st

import eit_core as core

st.set_page_config(page_title="EIT Analyzer Lambda V4.2.1", layout="wide")
st.title(f"EIT ANALYZER Λ — V{core.APP_VERSION}")
st.caption("Research prototype | Synthetic signals | Neural vs. classical denoising | Software-only visualization")


@st.cache_resource(show_spinner="Training neural network on synthetic signals...")
def get_model():
    return core.train_model()


model, info = get_model()
if not info.converged:
    st.warning("Model training did not converge; results are less reliable.")

st.sidebar.header("Experiment settings")
noise_std = st.sidebar.slider("Noise standard deviation", 0.05, 0.8, 0.35, 0.05)
seed = int(st.sidebar.number_input("Test seed", min_value=0, max_value=100000, value=999))
mode = st.sidebar.radio("Input", ["Generated signal", "Upload CSV/TXT"])
sig_rng, noise_rng = core.seeds_for(seed)

if mode == "Upload CSV/TXT":
    uploaded = st.sidebar.file_uploader(f"Exactly {core.N} numbers, '.' decimals, |x| ≤ {core.MAX_ABS_VALUE:g}, no header",
                                        type=["csv", "txt"])
    if not uploaded:
        st.info("Upload a numeric signal to continue.")
        st.stop()
    try:
        original, notes = core.parse_upload(uploaded.getvalue())
    except core.UploadError as exc:
        st.error(f"Rejected: {exc}")
        st.stop()
    for n in notes:
        st.warning(n)
    st.info("Your uploaded values are treated as the clean reference; synthetic noise is added on top. "
            "If your data is already a noisy measurement, the error numbers do not describe its true error.")
    data_type = "user_uploaded_treated_as_reference"
else:
    original = core.clean_signal(sig_rng)
    data_type = "synthetic"

res = core.run_experiment(model, original, float(noise_std), noise_rng)
m = res["metrics"]
for n in res["notes"]:
    st.warning(n)

c = st.columns(4)
c[0].metric("MSE — Noisy", "n/a" if m["mse_noisy"] is None else f"{m['mse_noisy']:.6f}")
c[1].metric("Neural reduction", core.format_pct(m["reduction_pct_neural"]))
c[2].metric("Moving avg (5) reduction", core.format_pct(m["reduction_pct_moving_average_5"]))
c[3].metric("FFT low-pass reduction", core.format_pct(m["reduction_pct_fft_lowpass_10"]))
if any(core.is_extreme_pct(m[f"reduction_pct_{k}"]) for k in res["recon"]):
    st.warning(f"At least one reduction exceeds ±{core.EXTREME_PCT_DISPLAY:g}% and is shown in scientific notation. "
               "This input is far outside the training scale; the exported values are unchanged.")
if m["reduction_pct_neural"] is not None and m["reduction_pct_neural"] < 0:
    st.warning("On this test the neural model is worse than the noisy input. This is a valid experimental result.")
if m["best_method"] is not None and not m["neural_beats_all_baselines"]:
    st.warning(f"A simple classical method ({m['best_method']}) did better than the neural model on this test.")

st.subheader("Original → Noise → Reconstructions")
fig = go.Figure()
series = [("Original", original), ("Noisy", res["noisy"])] + [(k, v) for k, v in res["recon"].items()]
for label, values in series:
    fig.add_trace(go.Scatter(x=list(range(core.N)), y=values, name=label, mode="lines"))
fig.update_layout(xaxis_title="Sample index", yaxis_title="Amplitude", height=420)
st.plotly_chart(fig, width="stretch")

st.subheader("3D visualization — decorative only")
st.caption("Sphere deformed by the neural error size. Not a hologram, not a measured field, carries no physical meaning.")
phi = np.linspace(0, 2 * np.pi, 60); theta = np.linspace(0, np.pi, 35)
pp, tt = np.meshgrid(phi, theta)
amp = 0.35 if m["mse_neural"] is None else min(0.35, np.sqrt(m["mse_neural"]) / 3)
r = 1 + amp * np.sin(5 * pp) * np.sin(3 * tt)
holo = go.Figure(go.Surface(x=r * np.sin(tt) * np.cos(pp), y=r * np.sin(tt) * np.sin(pp), z=r * np.cos(tt),
                            surfacecolor=np.sin(3 * pp), colorscale="Viridis", showscale=False, opacity=0.85))
holo.update_layout(height=420, scene=dict(xaxis_visible=False, yaxis_visible=False, zaxis_visible=False),
                   margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(holo, width="stretch")

record = {
    "app_version": core.APP_VERSION,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "data_type": data_type, "seed": seed, "noise_std": float(noise_std),
    "metrics": m, "model": core.model_info_dict(info),
    "environment": {"python": platform.python_version(), "numpy": np.__version__, "sklearn": sklearn.__version__},
    "boundary": "Software denoising experiment only. Not evidence for any EIT physical law.",
}

st.subheader("Session memory (lost on server restart)")
if "history" not in st.session_state:
    st.session_state.history = []
if st.button("Save this experiment in session memory"):
    st.session_state.history.append(record)
    st.success("Saved in session memory (not permanent storage).")
st.write(f"Session experiments: {len(st.session_state.history)}")
if st.session_state.history:
    st.dataframe(pd.DataFrame([{"timestamp_utc": h["timestamp_utc"], "seed": h["seed"], "noise_std": h["noise_std"],
                                 **{k: v for k, v in h["metrics"].items()}} for h in st.session_state.history]),
                 width="stretch")

st.subheader("Export")
df = pd.DataFrame({"index": np.arange(core.N), "original": original, "noisy": res["noisy"],
                   **{f"recon_{k}": v for k, v in res["recon"].items()}})
st.download_button("Download signal CSV", df.to_csv(index=False), "eit_v4_2_signal.csv", "text/csv")
try:
    st.download_button("Download experiment JSON", core.to_strict_json(record, indent=2),
                       "eit_v4_2_experiment.json", "application/json")
    if st.session_state.history:
        st.download_button("Download session memory JSON", core.to_strict_json(st.session_state.history, indent=2),
                           "eit_v4_2_memory.json", "application/json")
except core.ExportError as exc:
    st.error(f"JSON export unavailable: {exc}")

st.caption("Research boundary: neural signal reconstruction does not establish time travel, consciousness transfer, or "
           "a physical intelligence-dependent time law. Λ is a conceptual EIT label, not a measured physical constant.")
