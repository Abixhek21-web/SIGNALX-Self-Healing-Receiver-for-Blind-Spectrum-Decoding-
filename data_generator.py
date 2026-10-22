"""
Synthetic RF Signal and Controlled Transmitter Chain Generator.
Generates complete ground-truth transmitter chains:
Random bits -> FEC -> Interleaving -> Modulation -> Pulse Shaping -> CFO/Phase -> Timing -> AWGN -> IQ
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import signal as sp_signal

from synchronization.matched_filter import rrc_taps
from validation.crc import compute_crc16_ccitt
from decoding.convolutional import ViterbiDecoder
from decoding.reed_solomon import ReedSolomonCode
from deinterleaving.block import BlockDeinterleaver
from deinterleaving.convolutional import ConvolutionalDeinterleaver
from deinterleaving.diagonal import DiagonalDeinterleaver
from deinterleaving.pseudorandom import PseudoRandomDeinterleaver
from utils.logging import logger


SUPPORTED_MODS = ["BPSK", "QPSK", "8PSK", "16-QAM", "2-FSK"]


def generate_symbols(
    modulation: str,
    num_symbols: int,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate constellation symbols and bit sequence."""
    if seed is not None:
        np.random.seed(seed)

    mod_upper = modulation.upper()
    sync_bytes = bytes.fromhex("EB9055AA")
    sync_bits = np.unpackbits(np.frombuffer(sync_bytes, dtype=np.uint8))

    if "BPSK" in mod_upper:
        payload_bits = np.random.randint(0, 2, size=max(0, num_symbols - len(sync_bits)), dtype=np.uint8)
        bits = np.concatenate([sync_bits, payload_bits])[:num_symbols]
        symbols = (2.0 * bits.astype(float) - 1.0).astype(np.complex64)
    elif "8PSK" in mod_upper:
        total_bits = num_symbols * 3
        payload_bits = np.random.randint(0, 2, size=max(0, total_bits - len(sync_bits)), dtype=np.uint8)
        bits = np.concatenate([sync_bits, payload_bits])[:total_bits]
        triplets = bits.reshape(-1, 3)
        sectors = triplets[:, 0] * 4 + triplets[:, 1] * 2 + triplets[:, 2]
        symbols = np.exp(1j * sectors * (np.pi / 4.0)).astype(np.complex64)
    elif "16-QAM" in mod_upper or "QAM" in mod_upper:
        total_bits = num_symbols * 4
        payload_bits = np.random.randint(0, 2, size=max(0, total_bits - len(sync_bits)), dtype=np.uint8)
        bits = np.concatenate([sync_bits, payload_bits])[:total_bits]
        b = bits.reshape(-1, 4)
        pam4_lut = np.array([-3.0, -1.0, 3.0, 1.0])
        i_idx = b[:, 0] * 2 + b[:, 1]
        q_idx = b[:, 2] * 2 + b[:, 3]
        symbols = (pam4_lut[i_idx] + 1j * pam4_lut[q_idx]).astype(np.complex64) / np.sqrt(10.0)
    elif "FSK" in mod_upper:
        payload_bits = np.random.randint(0, 2, size=max(0, num_symbols - len(sync_bits)), dtype=np.uint8)
        bits = np.concatenate([sync_bits, payload_bits])[:num_symbols]
        symbols = (2.0 * bits.astype(float) - 1.0).astype(np.complex64)
    else:  # QPSK
        total_bits = num_symbols * 2
        payload_bits = np.random.randint(0, 2, size=max(0, total_bits - len(sync_bits)), dtype=np.uint8)
        bits = np.concatenate([sync_bits, payload_bits])[:total_bits]
        b_i = bits[0::2]
        b_q = bits[1::2]
        symbols = ((np.where(b_i == 0, 1.0, -1.0) + 1j * np.where(b_q == 0, 1.0, -1.0)) / np.sqrt(2.0)).astype(np.complex64)

    return symbols, bits


def generate_transmitter_chain(
    modulation: str = "QPSK",
    num_info_bits: int = 500,
    fec_type: str = "CONVOLUTIONAL_VITERBI",
    interleaver_type: str = "BLOCK",
    sample_rate: float = 1e6,
    symbol_rate: float = 250000.0,
    snr_db: float = 25.0,
    cfo_hz: float = 5000.0,
    phase_offset_rad: float = 0.35,
    rrc_beta: float = 0.35,
    timing_offset_samples: float = 0.0,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Generate complete ground-truth RF transmitter chain:
    Bits -> FEC -> Interleaver -> Modulation -> Pulse Shaping -> Channel Impairments -> IQ Samples.
    """
    if seed is not None:
        np.random.seed(seed)

    # 1. Generate Info Bits with known frame sync marker (0xEB90) and optional CRC
    sync_bytes = bytes.fromhex("EB90")
    sync_bits = np.unpackbits(np.frombuffer(sync_bytes, dtype=np.uint8))
    
    raw_payload = np.random.randint(0, 2, size=max(16, num_info_bits - len(sync_bits) - 16), dtype=np.uint8)
    # Compute CRC-16 on payload
    payload_bytes = np.packbits(raw_payload).tobytes()
    crc_val = compute_crc16_ccitt(payload_bytes)
    crc_bytes = crc_val.to_bytes(2, byteorder="big")
    crc_bits = np.unpackbits(np.frombuffer(crc_bytes, dtype=np.uint8))

    info_bits = np.concatenate([sync_bits, raw_payload, crc_bits])

    # 2. Forward Error Correction (FEC)
    fec_upper = fec_type.upper()
    if "CONVOLUTIONAL" in fec_upper or "VITERBI" in fec_upper:
        viterbi = ViterbiDecoder()
        fec_bits = viterbi.encode(info_bits)
    elif "REED_SOLOMON" in fec_upper or "RS" in fec_upper:
        rs = ReedSolomonCode(n=15, k=11)
        # Convert bits to GF(16) symbols
        n_nibbles = int(np.ceil(len(info_bits) / 4.0))
        padded_info = np.pad(info_bits, (0, n_nibbles * 4 - len(info_bits)))
        syms = [int(np.packbits(np.pad(padded_info[i*4:(i+1)*4], (4,0)))[0] >> 4) for i in range(n_nibbles)]
        # Encode blocks of 11 symbols
        encoded_syms = []
        for i in range(0, len(syms), 11):
            block = syms[i:i+11]
            enc_block = rs.encode(block)
            encoded_syms.extend(enc_block)
        # Convert symbols back to bits
        fec_bits_list = []
        for s in encoded_syms:
            fec_bits_list.extend([(s >> 3) & 1, (s >> 2) & 1, (s >> 1) & 1, s & 1])
        fec_bits = np.array(fec_bits_list, dtype=np.uint8)
    else:  # NO_FEC
        fec_bits = np.copy(info_bits)

    # 3. Interleaving
    intl_upper = interleaver_type.upper()
    if "BLOCK" in intl_upper:
        deint = BlockDeinterleaver(rows=4, cols=max(1, len(fec_bits) // 4))
        # Forward interleaving is writing row-by-row, reading col-by-col
        intlv_bits = deint.interleave(fec_bits)
    elif "CONVOLUTIONAL" in intl_upper:
        deint = ConvolutionalDeinterleaver(branches=4, delay_depth=2)
        intlv_bits = deint.interleave(fec_bits)
    elif "DIAGONAL" in intl_upper:
        deint = DiagonalDeinterleaver(block_size=16)
        intlv_bits = deint.interleave(fec_bits)
    elif "PSEUDO" in intl_upper:
        deint = PseudoRandomDeinterleaver(length=len(fec_bits), seed=seed or 42)
        intlv_bits = deint.interleave(fec_bits)
    else:  # PASS_THROUGH / NONE
        intlv_bits = np.copy(fec_bits)

    # 4. Digital Modulation Mapping
    mod_upper = modulation.upper()
    if "BPSK" in mod_upper:
        tx_symbols = (2.0 * intlv_bits.astype(float) - 1.0).astype(np.complex64)
    elif "8PSK" in mod_upper:
        pad_len = (3 - len(intlv_bits) % 3) % 3
        bits_pad = np.pad(intlv_bits, (0, pad_len))
        triplets = bits_pad.reshape(-1, 3)
        sectors = triplets[:, 0] * 4 + triplets[:, 1] * 2 + triplets[:, 2]
        tx_symbols = np.exp(1j * sectors * (np.pi / 4.0)).astype(np.complex64)
    elif "16-QAM" in mod_upper or "QAM" in mod_upper:
        pad_len = (4 - len(intlv_bits) % 4) % 4
        bits_pad = np.pad(intlv_bits, (0, pad_len))
        b = bits_pad.reshape(-1, 4)
        pam4_lut = np.array([-3.0, -1.0, 3.0, 1.0])
        i_idx = b[:, 0] * 2 + b[:, 1]
        q_idx = b[:, 2] * 2 + b[:, 3]
        tx_symbols = (pam4_lut[i_idx] + 1j * pam4_lut[q_idx]).astype(np.complex64) / np.sqrt(10.0)
    elif "FSK" in mod_upper:
        tx_symbols = (2.0 * intlv_bits.astype(float) - 1.0).astype(np.complex64)
    else:  # QPSK
        pad_len = (2 - len(intlv_bits) % 2) % 2
        bits_pad = np.pad(intlv_bits, (0, pad_len))
        b_i = bits_pad[0::2]
        b_q = bits_pad[1::2]
        i_sym = np.where(b_i == 0, 1.0, -1.0) / np.sqrt(2.0)
        q_sym = np.where(b_q == 0, 1.0, -1.0) / np.sqrt(2.0)
        tx_symbols = (i_sym + 1j * q_sym).astype(np.complex64)

    # 5. Pulse Shaping (RRC) & Upsampling
    sps = int(round(sample_rate / symbol_rate))
    sps = max(2, sps)

    if "FSK" in mod_upper:
        freq_dev = symbol_rate / 2.0
        f_inst = np.repeat(np.real(tx_symbols) * freq_dev, sps)
        phase = 2.0 * np.pi * np.cumsum(f_inst) / sample_rate + phase_offset_rad
        tx_signal = np.exp(1j * phase).astype(np.complex64)
    else:
        upsampled = np.zeros(len(tx_symbols) * sps, dtype=np.complex64)
        upsampled[::sps] = tx_symbols
        h_rrc = rrc_taps(beta=rrc_beta, sps=sps, num_symbols=8)
        tx_real = sp_signal.convolve(np.real(upsampled), h_rrc, mode="same")
        tx_imag = sp_signal.convolve(np.imag(upsampled), h_rrc, mode="same")
        tx_signal = (tx_real + 1j * tx_imag).astype(np.complex64)

    # 6. CFO & Phase Rotation
    n = np.arange(len(tx_signal))
    cfo_rotator = np.exp(1j * (2.0 * np.pi * cfo_hz * n / sample_rate + phase_offset_rad))
    signal_cfo = tx_signal * cfo_rotator

    # 7. Timing offset
    if abs(timing_offset_samples) > 1e-4:
        shift_samples = int(timing_offset_samples)
        if shift_samples > 0:
            signal_cfo = np.pad(signal_cfo, (shift_samples, 0))[:-shift_samples]

    # 8. AWGN Channel
    sig_power = np.mean(np.abs(signal_cfo) ** 2)
    snr_lin = 10.0 ** (snr_db / 10.0)
    noise_power = sig_power / snr_lin if snr_lin > 0 else 0.1
    noise_std = np.sqrt(noise_power / 2.0)
    noise = (np.random.normal(0, noise_std, len(signal_cfo)) +
             1j * np.random.normal(0, noise_std, len(signal_cfo))).astype(np.complex64)

    rx_samples = (signal_cfo + noise).astype(np.complex64)

    ground_truth = {
        "modulation": modulation,
        "fec_type": fec_type,
        "interleaver_type": interleaver_type,
        "sample_rate_hz": sample_rate,
        "symbol_rate_sps": symbol_rate,
        "sps": sps,
        "snr_db": snr_db,
        "cfo_hz": cfo_hz,
        "phase_offset_rad": phase_offset_rad,
        "rrc_beta": rrc_beta,
        "ground_truth_info_bits": info_bits,
        "ground_truth_fec_bits": fec_bits,
        "ground_truth_interleaved_bits": intlv_bits,
        "ground_truth_symbols": tx_symbols
    }

    return rx_samples, ground_truth


def generate_signal(
    modulation: str = "QPSK",
    num_symbols: int = 1000,
    sample_rate: float = 1e6,
    symbol_rate: float = 250000.0,
    snr_db: float = 20.0,
    cfo_hz: float = 5000.0,
    phase_offset_rad: float = 0.35,
    rrc_beta: float = 0.35,
    timing_offset_samples: float = 0.0,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, Dict[str, any]]:
    """Legacy helper wrapping generate_transmitter_chain."""
    rx_samples, gt = generate_transmitter_chain(
        modulation=modulation,
        num_info_bits=num_symbols * 2,
        fec_type="NO_FEC",
        interleaver_type="NONE",
        sample_rate=sample_rate,
        symbol_rate=symbol_rate,
        snr_db=snr_db,
        cfo_hz=cfo_hz,
        phase_offset_rad=phase_offset_rad,
        rrc_beta=rrc_beta,
        timing_offset_samples=timing_offset_samples,
        seed=seed
    )
    return rx_samples, gt


def generate_mixed_burst_signal(
    sample_rate: float = 1e6,
    seed: Optional[int] = 42
) -> Tuple[np.ndarray, List[Dict[str, any]]]:
    """Generate multi-burst mixed signal recording."""
    if seed is not None:
        np.random.seed(seed)

    noise_lead = (np.random.normal(0, 0.05, 2000) + 1j * np.random.normal(0, 0.05, 2000)).astype(np.complex64)

    b1_samples, meta1 = generate_transmitter_chain(
        modulation="QPSK",
        num_info_bits=600,
        fec_type="CONVOLUTIONAL_VITERBI",
        interleaver_type="BLOCK",
        sample_rate=sample_rate,
        symbol_rate=200000.0,
        snr_db=22.0,
        cfo_hz=8000.0,
        seed=101
    )

    noise_gap = (np.random.normal(0, 0.05, 1500) + 1j * np.random.normal(0, 0.05, 1500)).astype(np.complex64)

    b2_samples, meta2 = generate_transmitter_chain(
        modulation="16-QAM",
        num_info_bits=800,
        fec_type="NO_FEC",
        interleaver_type="NONE",
        sample_rate=sample_rate,
        symbol_rate=250000.0,
        snr_db=25.0,
        cfo_hz=-5000.0,
        seed=202
    )

    noise_tail = (np.random.normal(0, 0.05, 2000) + 1j * np.random.normal(0, 0.05, 2000)).astype(np.complex64)
    mixed = np.concatenate([noise_lead, b1_samples, noise_gap, b2_samples, noise_tail])

    meta = [
        {"segment": 1, "start_sample": len(noise_lead), "end_sample": len(noise_lead) + len(b1_samples), **meta1},
        {"segment": 2, "start_sample": len(noise_lead) + len(b1_samples) + len(noise_gap), "end_sample": len(noise_lead) + len(b1_samples) + len(noise_gap) + len(b2_samples), **meta2}
    ]

    return mixed, meta


def save_iq_to_file(samples: np.ndarray, file_path: str, dtype: str = "float32"):
    """Save complex samples as raw binary IQ file."""
    p = Path(file_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        if dtype == "float32":
            data = np.empty(len(samples) * 2, dtype=np.float32)
            data[0::2] = np.real(samples)
            data[1::2] = np.imag(samples)
            data.tofile(f)
        elif dtype == "int16":
            peak = np.max(np.abs(samples))
            norm = samples / (peak + 1e-12) * 32000.0
            data = np.empty(len(samples) * 2, dtype=np.int16)
            data[0::2] = np.real(norm).astype(np.int16)
            data[1::2] = np.imag(norm).astype(np.int16)
            data.tofile(f)
        elif dtype == "npy":
            np.save(p, samples)
    logger.info(f"Saved synthetic IQ signal to: {file_path}")
