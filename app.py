"""
RF Signal Analysis and Parameter Extraction - Streamlit Dashboard.
Production-grade interactive intelligence platform for authorized SDR recordings (.IQ, .WAV, .SigMF).
Fully compliant with SIH 2026 15-Stage Signal Processing Pipeline Architecture.
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px

from ingestion import load_signal, SignalData
from pipeline import SignalAnalysisPipeline, AnalysisResult, AnalysisContext
from data_generator import (
    generate_signal,
    generate_transmitter_chain,
    generate_mixed_burst_signal,
    save_iq_to_file,
    SUPPORTED_MODS
)
from visualization import (
    create_waveform_plot,
    create_psd_plot,
    create_constellation_plot,
    create_sync_comparison_plot,
    create_polar_distribution_plot,
    create_spectrogram_plot,
    generate_markdown_report
)
from utils.config import load_config
from utils.logging import logger


# Configure Streamlit Page
st.set_page_config(
    page_title="RF Signal Analyzer & Parameter Extraction",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Sleek Modern RF Intelligence Dashboard
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #00D2FF 0%, #3A7BD5 50%, #00F2FE 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        color: #9EACB8;
        font-size: 1.0rem;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 15px;
        text-align: center;
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #00F0FF;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #8C9BA5;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 6px;
        overflow-x: auto;
    }
    .stTabs [data-baseweb="tab"] {
        height: 40px;
        white-space: nowrap;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-pass {
        background-color: #00E676;
        color: black;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .badge-fail {
        background-color: #FF5252;
        color: white;
        font-weight: bold;
        padding: 4px 8px;
        border-radius: 4px;
    }
    .badge-na {
        background-color: #78909C;
        color: white;
        padding: 4px 8px;
        border-radius: 4px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_pipeline():
    return SignalAnalysisPipeline()


pipeline = get_pipeline()


# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite-sending-signal.png", width=64)
    st.markdown("### **Signal Ingestion & Sources**")

    source_type = st.radio(
        "Signal Source",
        ["Synthetic Generator", "Sample Signal Files", "Upload File (.iq, .wav, .sigmf)"]
    )

    signal_input_data = None
    input_sample_rate = 1e6
    center_freq_input = 100e6

    if source_type == "Synthetic Generator":
        st.markdown("#### **Transmitter Chain Configuration**")
        synth_mod = st.selectbox("Modulation", SUPPORTED_MODS, index=1)
        synth_fec = st.selectbox("FEC Coding", ["NO_FEC", "CONVOLUTIONAL_VITERBI", "REED_SOLOMON", "LDPC"], index=1)
        synth_intl = st.selectbox("Interleaver", ["BLOCK", "CONVOLUTIONAL", "DIAGONAL", "PSEUDORANDOM", "NONE"], index=0)
        synth_snr = st.slider("SNR (dB)", min_value=0.0, max_value=40.0, value=25.0, step=1.0)
        synth_cfo = st.slider("CFO (Hz)", min_value=-20000.0, max_value=20000.0, value=4500.0, step=500.0)
        synth_sps = st.selectbox("Samples per Symbol (Sps)", [2, 4, 8], index=1)
        synth_symbols = st.slider("Number of Info Bits", min_value=100, max_value=2000, value=400, step=50)

        synth_sr = 1e6
        synth_sym_rate = synth_sr / synth_sps
        input_sample_rate = synth_sr

        if st.button("🚀 Generate & Analyze", type="primary"):
            rx_samples, gt = generate_transmitter_chain(
                modulation=synth_mod,
                num_info_bits=synth_symbols,
                fec_type=synth_fec,
                interleaver_type=synth_intl,
                sample_rate=synth_sr,
                symbol_rate=synth_sym_rate,
                snr_db=synth_snr,
                cfo_hz=synth_cfo,
                seed=int(time.time()) % 1000
            )
            signal_input_data = SignalData(
                samples=rx_samples,
                sample_rate=synth_sr,
                center_frequency=center_freq_input,
                metadata={"format": "synthetic_ground_truth", "ground_truth": gt},
                source_file=f"synthetic_{synth_mod.lower()}_{synth_fec.lower()}.iq"
            )

    elif source_type == "Sample Signal Files":
        sample_dir = PROJECT_ROOT / "data" / "sample_signals"
        if sample_dir.exists():
            files = [f.name for f in sample_dir.glob("*.*") if not f.name.endswith(".txt")]
        else:
            files = []

        if files:
            selected_file = st.selectbox("Select Sample Signal", files)
            file_path = str(sample_dir / selected_file)
            input_sample_rate = st.number_input("Sampling Rate (Hz)", value=1000000.0, step=100000.0)
            center_freq_input = st.number_input("Center Frequency (Hz)", value=100000000.0, step=1000000.0)

            if st.button("🚀 Load & Analyze", type="primary"):
                signal_input_data = file_path
        else:
            st.warning("No sample files found in data/sample_signals/.")

    else:
        uploaded_file = st.file_uploader("Upload IQ, WAV, or SigMF file", type=["iq", "raw", "bin", "wav", "npy"])
        input_sample_rate = st.number_input("Sampling Rate (Hz)", value=1000000.0, step=100000.0)
        center_freq_input = st.number_input("Center Frequency (Hz)", value=100000000.0, step=1000000.0)

        if uploaded_file is not None and st.button("🚀 Analyze Uploaded File", type="primary"):
            temp_path = PROJECT_ROOT / "temp_upload"
            temp_path.mkdir(exist_ok=True)
            saved_file = temp_path / uploaded_file.name
            with open(saved_file, "wb") as f:
                f.write(uploaded_file.getbuffer())
            signal_input_data = str(saved_file)


# Header Title
st.markdown('<div class="main-title">📡 RF Signal Analysis & Parameter Extraction Engine</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">SIH 2026 End-to-End Automated SDR Demodulation, De-interleaving, Decoding & Structural Intelligence Platform</div>', unsafe_allow_html=True)


# Run analysis if input triggered or maintain session state
if signal_input_data is not None:
    with st.spinner("Executing 15-Stage Pipeline Analysis..."):
        res, ctx = pipeline.run(
            signal_input=signal_input_data,
            sample_rate=input_sample_rate,
            center_frequency=center_freq_input
        )
        st.session_state["result"] = res
        st.session_state["context"] = ctx

if "result" not in st.session_state:
    # Default initial analysis on clean QPSK signal
    with st.spinner("Initializing default signal analysis..."):
        samples, gt = generate_transmitter_chain(
            modulation="QPSK",
            num_info_bits=400,
            fec_type="CONVOLUTIONAL_VITERBI",
            interleaver_type="BLOCK",
            sample_rate=1e6,
            symbol_rate=250000.0,
            snr_db=25.0,
            cfo_hz=4000.0,
            seed=42
        )
        default_sig = SignalData(
            samples=samples,
            sample_rate=1e6,
            center_frequency=100e6,
            metadata={"format": "synthetic_ground_truth", "ground_truth": gt},
            source_file="demo_qpsk_viterbi.iq"
        )
        res, ctx = pipeline.run(default_sig)
        st.session_state["result"] = res
        st.session_state["context"] = ctx

result: AnalysisResult = st.session_state["result"]
ctx: AnalysisContext = st.session_state["context"]


# --- 15 PIPELINE TABS ---
tab_names = [
    "1. Overview",
    "2. Ingestion",
    "3. Discovery",
    "4. Segmentation",
    "5. I/Q Geometry",
    "6. Parameter ID",
    "7. Estimation",
    "8. Synchronization",
    "9. Demodulation",
    "10. De-interleaving",
    "11. FEC Decoding",
    "12. Bit Correlation",
    "13. Validation",
    "14. Adaptive Search",
    "15. Final Report"
]

tabs = st.tabs(tab_names)


# --- TAB 1: OVERVIEW ---
with tabs[0]:
    st.markdown("### 📊 Pipeline Executive Overview")
    
    # Key Metric Cards
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Identified Modulation", result.top_hypothesis.modulation, f"Rank #{result.top_hypothesis.rank}")
    with c2:
        st.metric("Combined Confidence", f"{result.top_hypothesis.combined_confidence:.1%}")
    with c3:
        st.metric("Symbol Rate", f"{result.parameters.symbol_rate_sps / 1e3:.1f} kSps")
    with c4:
        st.metric("In-Band SNR", f"{result.parameters.snr_db:.1f} dB")
    with c5:
        val_badge = "✅ PASS" if result.validation_result.passed else "❌ FAIL"
        st.metric("Structural Validation", val_badge, f"{result.validation_result.score:.1f}/100")

    col_l, col_r = st.columns([1.2, 1])
    with col_l:
        st.markdown("#### **Signal Spectrum & Occupied Bandwidth**")
        if ctx.discovery_result:
            fig_psd = create_psd_plot(
                ctx.discovery_result.psd_freqs,
                ctx.discovery_result.psd_power,
                center_freq=result.signal_data.center_frequency or 0.0,
                obw_hz=result.parameters.bandwidth_hz
            )
            st.plotly_chart(fig_psd, use_container_width=True)
    with col_r:
        st.markdown("#### **Recovered Constellation Cloud**")
        fig_const = create_constellation_plot(
            result.sync_result.synced_symbols,
            modulation_name=result.top_hypothesis.modulation,
            evm_rms_pct=result.demod_result.evm_rms_pct
        )
        st.plotly_chart(fig_const, use_container_width=True)


# --- TAB 2: SIGNAL INGESTION ---
with tabs[1]:
    st.markdown("### 📥 Stage 1: Signal Ingestion & Metadata Extraction")
    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("#### **Extracted Signal Metadata**")
        meta_table = {
            "Source File": result.signal_data.source_file,
            "Format": result.signal_data.format,
            "Total Samples": f"{result.signal_data.num_samples:,}",
            "Sample Rate": f"{result.signal_data.sample_rate / 1e6:.3f} MSps",
            "Center Frequency": f"{result.signal_data.center_frequency / 1e6:.3f} MHz" if result.signal_data.center_frequency else "UNKNOWN",
            "Duration": f"{result.signal_data.duration:.4f} s",
            "Data Type": "complex64",
            "DC Offset Removal": "Applied",
            "Power Normalization": "0 dBFS Normalized"
        }
        st.dataframe(pd.DataFrame(list(meta_table.items()), columns=["Field", "Value"]), use_container_width=True)

    with col2:
        st.markdown("#### **Raw vs. Normalized Time-Domain Waveform**")
        raw_preview = result.signal_data.raw_samples[:1000] if result.signal_data.raw_samples is not None else result.signal_data.samples[:1000]
        norm_preview = result.signal_data.samples[:1000]
        fig_wave = create_waveform_plot(norm_preview, sample_rate=result.signal_data.sample_rate)
        st.plotly_chart(fig_wave, use_container_width=True)


# --- TAB 3: SIGNAL DISCOVERY ---
with tabs[2]:
    st.markdown("### 🔍 Stage 2: Signal Discovery & Spectral Analysis")
    if ctx.discovery_result:
        d1, d2, d3, d4 = st.columns(4)
        with d1:
            st.metric("Carrier Frequency Offset", f"{ctx.discovery_result.carrier_frequency:+.1f} Hz")
        with d2:
            st.metric("Occupied Bandwidth (99%)", f"{ctx.discovery_result.bandwidth / 1e3:.1f} kHz")
        with d3:
            st.metric("Estimated Noise Floor", f"{ctx.discovery_result.noise_floor_db:.1f} dBFS")
        with d4:
            st.metric("Estimated SNR", f"{ctx.discovery_result.estimated_snr:.1f} dB")

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### **Welch Power Spectral Density**")
            fig_psd = create_psd_plot(ctx.discovery_result.psd_freqs, ctx.discovery_result.psd_power)
            st.plotly_chart(fig_psd, use_container_width=True)
        with col_s2:
            st.markdown("#### **STFT Time-Frequency Spectrogram**")
            if ctx.discovery_result.spectrogram_t is not None:
                fig_spec = create_spectrogram_plot(
                    ctx.discovery_result.spectrogram_t,
                    ctx.discovery_result.spectrogram_f,
                    ctx.discovery_result.spectrogram_sxx
                )
                st.plotly_chart(fig_spec, use_container_width=True)


# --- TAB 4: SIGNAL SEGMENTATION ---
with tabs[3]:
    st.markdown("### ✂️ Stage 3: Energy-Based Signal Segmentation")
    st.write(f"Detected **{len(result.segments)}** independent signal burst regions.")
    if result.segments:
        seg_rows = []
        for i, s in enumerate(result.segments):
            seg_rows.append({
                "Segment ID": i + 1,
                "Start Sample": s.start_sample,
                "End Sample": s.end_sample,
                "Duration (ms)": f"{s.duration_sec * 1e3:.2f}",
                "Occupied BW (kHz)": f"{s.bandwidth_hz / 1e3:.1f}",
                "In-Band SNR (dB)": f"{s.snr_db:.1f}"
            })
        st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)

        # Plot energy envelope
        env = np.abs(result.signal_data.samples[:10000])
        fig_env = go.Figure()
        fig_env.add_trace(go.Scatter(y=env, mode='lines', name='Envelope |z[n]|', line=dict(color='#00D2FF', width=1)))
        for s in result.segments:
            if s.end_sample <= 10000:
                fig_env.add_vrect(x0=s.start_sample, x1=s.end_sample, fillcolor="rgba(0,230,118,0.2)", layer="below", line_width=0)
        fig_env.update_layout(title="Signal Burst Regions and Segmentation Mask", template="plotly_dark")
        st.plotly_chart(fig_env, use_container_width=True)


# --- TAB 5: I/Q POINT-COORDINATE ENGINE ---
with tabs[4]:
    st.markdown("### 🔬 Stage 4: I/Q Point-Coordinate Geometry Engine")
    st.info("Analyzes point cloud coordinates $(I[n], Q[n])$, radial rings, cluster spread, and rotational symmetries.")
    
    iq_geom = result.iq_geometry
    g1, g2, g3, g4 = st.columns(4)
    with g1:
        st.metric("Centroid (I, Q)", f"({iq_geom.centroid_i:+.3f}, {iq_geom.centroid_q:+.3f})")
    with g2:
        st.metric("I/Q Covariance", f"{iq_geom.covariance_iq:+.5f}")
    with g3:
        st.metric("Estimated Radial Rings", f"{iq_geom.estimated_radial_rings}")
    with g4:
        st.metric("Estimated Clusters", f"{iq_geom.estimated_cluster_count}")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("#### **I/Q Point Cloud Scatter**")
        fig_iq = go.Figure()
        samp = result.active_segment.samples[:2000]
        fig_iq.add_trace(go.Scatter(x=np.real(samp), y=np.imag(samp), mode='markers', marker=dict(color='#00F0FF', size=3, opacity=0.6), name="I/Q Points"))
        fig_iq.update_layout(title="Complex Sample Point Coordinates", xaxis_title="In-Phase (I)", yaxis_title="Quadrature (Q)", template="plotly_dark")
        st.plotly_chart(fig_iq, use_container_width=True)

    with col_g2:
        st.markdown("#### **Rotational Symmetry Line Spectrum**")
        sym_df = pd.DataFrame([
            {"Symmetry Order": "2-Fold (BPSK line)", "Spectral Prominence": iq_geom.symmetry_order_2},
            {"Symmetry Order": "4-Fold (QPSK line)", "Spectral Prominence": iq_geom.symmetry_order_4},
            {"Symmetry Order": "8-Fold (8PSK line)", "Spectral Prominence": iq_geom.symmetry_order_8},
        ])
        fig_sym = px.bar(sym_df, x="Symmetry Order", y="Spectral Prominence", color="Spectral Prominence", template="plotly_dark", title="CFO-Invariant Harmonic Symmetry Order")
        st.plotly_chart(fig_sym, use_container_width=True)


# --- TAB 6: SIGNAL PARAMETER IDENTIFICATION ---
with tabs[5]:
    st.markdown("### 🧠 Stage 5: Hybrid Parameter Identification Engine (DSP + ML Fusion)")
    st.write("Combines mathematical cyclostationary/higher-order statistical scoring with tabular Random Forest ML inference.")
    
    hypo_rows = []
    for h in result.fused_hypotheses:
        hypo_rows.append({
            "Rank": h.rank,
            "Modulation": h.modulation,
            "Combined Confidence": f"{h.combined_confidence:.2%}",
            "DSP Mathematical Score": f"{h.dsp_score:.2%}",
            "ML Probability": f"{h.ml_score:.2%}",
            "Evidence / Reasoning": h.reasoning
        })
    st.dataframe(pd.DataFrame(hypo_rows), use_container_width=True)

    # Plot confidence comparison
    df_conf = pd.DataFrame([{"Modulation": h.modulation, "Combined Confidence": h.combined_confidence, "DSP Score": h.dsp_score, "ML Probability": h.ml_score} for h in result.fused_hypotheses])
    fig_fused = go.Figure()
    fig_fused.add_trace(go.Bar(name='Combined Confidence', x=df_conf['Modulation'], y=df_conf['Combined Confidence'], marker_color='#00F0FF'))
    fig_fused.add_trace(go.Bar(name='DSP Score', x=df_conf['Modulation'], y=df_conf['DSP Score'], marker_color='#3A7BD5'))
    fig_fused.add_trace(go.Bar(name='ML Probability', x=df_conf['Modulation'], y=df_conf['ML Probability'], marker_color='#9C27B0'))
    fig_fused.update_layout(barmode='group', title="Top-K Hypotheses Multi-Evidence Breakdown", template="plotly_dark")
    st.plotly_chart(fig_fused, use_container_width=True)


# --- TAB 7: PARAMETER ESTIMATION ---
with tabs[6]:
    st.markdown("### 📐 Stage 6: Physical Parameter Estimation")
    p = result.parameters
    param_rows = [
        {"Parameter": "Carrier Frequency", "Estimated Value": f"{p.carrier_frequency_hz:+.1f} Hz", "Status": "ESTIMATED"},
        {"Parameter": "Sampling Rate", "Estimated Value": f"{result.signal_data.sample_rate / 1e6:.3f} MSps", "Status": "KNOWN (Ingestion)"},
        {"Parameter": "Symbol Rate (Baud)", "Estimated Value": f"{p.symbol_rate_sps:,.1f} Sps", "Status": "ESTIMATED (Squaring)"},
        {"Parameter": "Occupied Bandwidth (99%)", "Estimated Value": f"{p.bandwidth_hz / 1e3:.1f} kHz", "Status": "ESTIMATED (PSD)"},
        {"Parameter": "Estimated SNR", "Estimated Value": f"{p.snr_db:.2f} dB", "Status": "ESTIMATED"},
        {"Parameter": "Fine CFO", "Estimated Value": f"{p.cfo_hz:+.1f} Hz", "Status": "ESTIMATED"},
        {"Parameter": "Timing Offset", "Estimated Value": f"{p.timing_offset_samples:+.2f} samples", "Status": "ESTIMATED"},
        {"Parameter": "Carrier Phase Offset", "Estimated Value": f"{p.phase_offset_rad:+.3f} rad", "Status": "ESTIMATED"}
    ]
    st.dataframe(pd.DataFrame(param_rows), use_container_width=True)


# --- TAB 8: SYNCHRONIZATION ---
with tabs[7]:
    st.markdown("### 🔄 Stage 7: Synchronization Pipeline")
    sync = result.sync_result
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("CFO Corrected", f"{sync.cfo_corrected_hz:+.1f} Hz")
    with s2:
        st.metric("Samples per Symbol", f"{sync.samples_per_symbol:.2f}")
    with s3:
        st.metric("EVM Before Sync", f"{sync.evm_before_pct:.1f}%")
    with s4:
        st.metric("EVM After Sync", f"{sync.evm_after_pct:.1f}%", f"-{sync.evm_before_pct - sync.evm_after_pct:.1f}%")

    col_syn1, col_syn2 = st.columns(2)
    with col_syn1:
        fig_syn = create_sync_comparison_plot(sync.raw_samples, sync.synced_symbols, modulation_name=result.top_hypothesis.modulation)
        st.plotly_chart(fig_syn, use_container_width=True)
    with col_syn2:
        # Phase tracking history
        st.markdown("#### **Carrier Phase Tracking History**")
        fig_ph = go.Figure()
        fig_ph.add_trace(go.Scatter(y=sync.phase_history[:1000], mode='lines', line=dict(color='#00E676', width=1.5), name="Phase (rad)"))
        fig_ph.update_layout(title="Decision-Directed PLL Phase History", xaxis_title="Symbol Index", yaxis_title="Phase (radians)", template="plotly_dark")
        st.plotly_chart(fig_ph, use_container_width=True)


# --- TAB 9: DEMODULATION ---
with tabs[8]:
    st.markdown("### 🔀 Stage 8: Demodulation Engine")
    demod = result.demod_result
    st.write(f"Demodulated using **{demod.modulation}** demodulator with soft LLR extraction.")
    dm1, dm2, dm3 = st.columns(3)
    with dm1:
        st.metric("Demodulated Bits", f"{demod.bit_count:,}")
    with dm2:
        st.metric("Recovered Symbols", f"{demod.symbol_count:,}")
    with dm3:
        st.metric("Constellation EVM RMS", f"{demod.evm_rms_pct:.2f}%")

    st.markdown("#### **Extracted Hard Bitstream Preview**")
    st.code(demod.bit_string_preview or "None", language="text")
    st.markdown("#### **Hex Byte Stream Preview**")
    st.code(demod.hex_preview or "None", language="text")

    # Upgrade 6: LLR confidence display
    st.markdown("#### **Soft-Decision (LLR) Confidence Indicators**")
    llr_diag = demod.diagnostics or {}
    llr_nv = llr_diag.get("llr_noise_var", None)
    llr_scale = llr_diag.get("llr_scale", None)
    lc1, lc2, lc3 = st.columns(3)
    with lc1:
        if llr_nv is not None:
            st.metric("LLR Noise Variance (sigma^2)", f"{float(llr_nv):.4f}")
        else:
            st.metric("LLR Noise Variance (sigma^2)", "N/A")
    with lc2:
        if llr_scale is not None:
            st.metric("LLR Scale Factor (1/sigma^2)", f"{float(llr_scale):.2f}")
        else:
            snr = result.parameters.snr_db
            est_nv = 10 ** (-snr / 10) / 2
            st.metric("LLR Scale Factor (est.)", f"{1/est_nv:.2f}")
    with lc3:
        evm_pct = demod.evm_rms_pct
        evm_conf = max(0.0, 1.0 - evm_pct / 30.0)
        st.metric("EVM-Derived Confidence", f"{evm_conf:.1%}")
    st.caption(
        "LLR (Log-Likelihood Ratio) outputs are scaled by the estimated noise variance "
        "from Stage 3. Higher |LLR| = more reliable bit decisions. EVM-derived confidence "
        "is 1 - (EVM%/30), capped at 0."
    )


# --- TAB 10: DE-INTERLEAVING ---
with tabs[9]:
    st.markdown("### 🔀 Stage 9: De-interleaving Engine")
    deint = result.deinterleave_result
    st.write(f"Active De-interleaver Configuration: **{deint.interleaver_type}**")
    st.json(deint.parameters)
    st.write(f"Output De-interleaved Bit Count: **{len(deint.deinterleaved_bits):,} bits**")


# --- TAB 11: FEC DECODER BANK ---
with tabs[10]:
    st.markdown("### 🛡️ Stage 10: FEC Decoder Bank")
    dec = result.decoder_result
    st.write(f"Active Decoder Candidate: **{dec.decoder_name}**")
    fc1, fc2, fc3 = st.columns(3)
    with fc1:
        st.metric("Decoder Convergence", "SUCCESS" if dec.success else "FAILURE")
    with fc2:
        st.metric("Decoder Confidence", f"{dec.confidence:.2%}")
    with fc3:
        st.metric("Corrected Errors", f"{dec.corrected_errors}")

    st.markdown("#### **Decoded Payload Diagnostics**")
    st.json(dec.diagnostics)

    # Upgrade 6: Honest identification label display
    diag = dec.diagnostics or {}
    id_method = diag.get("identification_method", None)
    candidate_note = diag.get("candidate_library_match", None)
    if id_method:
        badge = "Evidence-Based" if id_method == "fused_dsp_ml_evidence" else "Library Match"
        st.info(
            f"**Identification Method:** {badge} (`{id_method}`)  \n"
            + (f"**Note:** {candidate_note}" if candidate_note else "")
        )


# --- TAB 12: BIT STREAM CORRELATION ---
with tabs[11]:
    st.markdown("### 🔁 Stage 11: Bit Stream Correlation & Preamble Engine")
    corr = result.correlation_result
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.metric("Correlation Score", f"{corr.correlation_score:.1f}/100")
    with bc2:
        st.metric("Matched Preamble", corr.preamble_match or "None", f"Offset: {corr.preamble_offset}" if corr.preamble_offset >= 0 else "")
    with bc3:
        st.metric("Detected Periodic Patterns", f"{len(corr.detected_patterns)}")

    st.markdown("#### **Detected Binary Patterns**")
    if corr.detected_patterns:
        st.dataframe(pd.DataFrame(corr.detected_patterns), use_container_width=True)
    else:
        st.info("No repetitive binary n-grams detected in payload.")

    # Upgrade 5+6: Frame boundary candidates display
    st.markdown("#### **Frame Boundary Detection (Upgrade 5)**")
    frame_cands = (corr.diagnostics or {}).get("frame_candidates", [])
    if frame_cands:
        frame_df = pd.DataFrame([
            {
                "Preamble": c["preamble_name"],
                "Frame Length (bits)": c["frame_length_bits"],
                "Repeats Found": c["hit_count"],
                "Confidence": f"{c['confidence']:.1%}",
                "Sample Offsets": str(c.get("hit_offsets", [])[:5])
            }
            for c in frame_cands
        ])
        st.dataframe(frame_df, use_container_width=True)
        best = frame_cands[0]
        st.success(
            f"Best frame candidate: **{best['preamble_name']}** "
            f"at **{best['frame_length_bits']} bits/frame** "
            f"({best['hit_count']} hits, confidence {best['confidence']:.1%})"
        )
    else:
        st.info("No regularly-spaced preamble frames detected in bitstream.")


# --- TAB 13: STRUCTURAL VALIDATION ---
with tabs[12]:
    st.markdown("### ✅ Stage 12: Structural Validation")
    val = result.validation_result
    vc1, vc2 = st.columns([1, 2])
    with vc1:
        st.metric("Overall Validation Status", val.valid, f"Score: {val.score:.1f}/100")
        st.markdown("#### **Validation Checks Breakdown**")
        checks_df = pd.DataFrame([{"Check Name": k, "Result": v} for k, v in val.checks.items()])
        st.dataframe(checks_df, use_container_width=True)
    with vc2:
        st.markdown("#### **Evidence & Validation Reasons**")
        for r in val.reasons:
            st.write(f"• {r}")
        if val.failed_checks:
            st.markdown("#### **Failed Checks**")
            for f in val.failed_checks:
                st.write(f"⚠️ {f}")


# --- TAB 14: ADAPTIVE SEARCH ---
with tabs[13]:
    st.markdown("### 🔄 Stage 13: Bounded Adaptive Hypothesis Search")
    st.write(f"Evaluated **{result.search_result.total_attempts}** candidate configurations across Top-K Modulations, De-interleavers, and FEC Decoders.")
    
    hist_rows = [a.to_dict() for a in result.search_result.history]
    st.dataframe(pd.DataFrame(hist_rows), use_container_width=True)


# --- TAB 15: FINAL REPORT ---
with tabs[14]:
    st.markdown("### 📄 Stage 14: Final Intelligence & Technical Report")
    report_md = generate_markdown_report(result)
    st.markdown(report_md)
    st.download_button(
        label="📥 Download Markdown Report",
        data=report_md,
        file_name="rf_signal_analysis_report.md",
        mime="text/markdown"
    )
    st.download_button(
        label="📥 Download Structured JSON",
        data=result.to_json(),
        file_name="rf_signal_analysis_result.json",
        mime="application/json"
    )

    # Upgrade 7: SigMF round-trip metadata export
    st.markdown("---")
    st.markdown("#### **📡 SigMF Metadata Export (Upgrade 7)**")
    st.write(
        "Export the pipeline analysis results as a SigMF-compliant `.sigmf-meta` file. "
        "The file uses the standard `core:` namespace plus a `sih2026:` extension namespace "
        "to carry all estimated parameters, identification method labels, and frame detection results."
    )
    try:
        from ingestion.sigmf_export import build_sigmf_meta
        sigmf_doc = build_sigmf_meta(
            source_meta=None,
            analysis_result=result,
            analysis_context=ctx if "ctx" in dir() else None,
        )
        sigmf_json_str = json.dumps(sigmf_doc, indent=2, default=str)
        st.download_button(
            label="📥 Download SigMF Metadata (.sigmf-meta)",
            data=sigmf_json_str,
            file_name="analysis_result.sigmf-meta",
            mime="application/json",
        )
        with st.expander("Preview SigMF Metadata"):
            st.json(sigmf_doc)
    except Exception as _sigmf_err:
        st.warning(f"SigMF export unavailable: {_sigmf_err}")
