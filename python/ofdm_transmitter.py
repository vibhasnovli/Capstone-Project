import numpy as np
import sys
import os

# Import our parameters and ZC generator
sys.path.append(os.path.dirname(__file__))
from ofdm_params import (
    FFT_SIZE, CP_LEN, NUM_PILOTS, NUM_SYMBOLS,
    PILOT_BINS, PILOT_VALUES,
    ZC_LENGTH, ZC_ROOT,
    SAMPLES_PER_PERIOD, SAMPLES_PER_BURST, GUARD_SAMPLES
)
from zc_preamble import generate_zc


# =============================================================================
# OFDM Symbol Generation
# =============================================================================

def generate_ofdm_symbol(pilot_values: np.ndarray) -> np.ndarray:
    """
    Generate one OFDM symbol with cyclic prefix.
    """
    # 1. Build frequency domain buffer
    freq_domain = np.zeros(FFT_SIZE, dtype=np.complex64)

    # 2. Place pilot values onto their assigned bins
    freq_domain[PILOT_BINS] = pilot_values

    # 3. IFFT
    time_domain = np.fft.ifft(freq_domain).astype(np.complex64)

    # 4. Prepend cyclic prefix (normalization now handled at burst level)
    cyclic_prefix = time_domain[-CP_LEN:]
    symbol_with_cp = np.concatenate([cyclic_prefix, time_domain])

    return symbol_with_cp  # shape: (80,)

# =============================================================================
# Full Burst Generation
# =============================================================================

def generate_burst() -> np.ndarray:
    """
    Generate one complete TX burst.

    Structure:
        [ ZC preamble (63) | OFDM symbol x8 (80 each) ]
        Total: 63 + 8*80 = 703 samples

    Returns
    -------
    np.ndarray
        Complex burst samples. Shape: (SAMPLES_PER_BURST,) = (703,)
    """
    # 1. ZC preamble
    zc = generate_zc(ZC_LENGTH, ZC_ROOT).astype(np.complex64)

    # 2. Generate all OFDM symbols WITHOUT per-symbol normalization
    ofdm_symbols = np.concatenate([
        generate_ofdm_symbol(PILOT_VALUES)
        for _ in range(NUM_SYMBOLS)
    ])

    # 3. Concatenate preamble + symbols
    burst = np.concatenate([zc, ofdm_symbols])

    # 4. Normalize entire burst to unit average power
    power = np.mean(np.abs(burst) ** 2)
    if power > 0:
        burst /= np.sqrt(power)

    # 5. Apply clipping to limit peak amplitude (reduces PAPR)
    clip_level = 3.0  # max amplitude — clips anything above 3x RMS
    burst_clipped = np.clip(burst.real, -clip_level, clip_level) + \
                    1j * np.clip(burst.imag, -clip_level, clip_level)

    assert len(burst_clipped) == SAMPLES_PER_BURST, \
        f"Burst length mismatch: got {len(burst_clipped)}, expected {SAMPLES_PER_BURST}"

    return burst_clipped.astype(np.complex64)


# =============================================================================
# Full TX Frame Generation (burst + guard zeros = one period)
# =============================================================================

def generate_tx_period() -> np.ndarray:
    """
    Generate one complete TX period = burst + zero padding.

    The zeros give the RX side silence between bursts, preventing
    inter-burst interference and giving the Pluto TX time to settle.

    Returns
    -------
    np.ndarray
        Complex samples for one period. Shape: (SAMPLES_PER_PERIOD,) = (200000,)
    """
    burst = generate_burst()
    guard = np.zeros(GUARD_SAMPLES, dtype=np.complex64)
    period = np.concatenate([burst, guard])

    assert len(period) == SAMPLES_PER_PERIOD, \
        f"Period length mismatch: got {len(period)}, expected {SAMPLES_PER_PERIOD}"

    return period


# =============================================================================
# Continuous TX Buffer (N repeated periods)
# =============================================================================

def generate_tx_buffer(num_periods: int = 10) -> np.ndarray:
    """
    Generate a buffer of N repeated TX periods for continuous transmission.

    Parameters
    ----------
    num_periods : int
        Number of periods to concatenate (default 10 = 100ms of TX)

    Returns
    -------
    np.ndarray
        Complex TX buffer. Shape: (num_periods * SAMPLES_PER_PERIOD,)
    """
    period = generate_tx_period()
    return np.tile(period, num_periods).astype(np.complex64)


# =============================================================================
# Verification
# =============================================================================

if __name__ == "__main__":
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    print("Generating burst...")
    burst = generate_burst()

    print("Generating TX period...")
    period = generate_tx_period()

    print("Generating TX buffer (10 periods)...")
    buffer = generate_tx_buffer(num_periods=10)

    # --- Print summary ---
    print("\n" + "=" * 45)
    print("        TX Signal Summary")
    print("=" * 45)
    print(f"  Burst length      : {len(burst)} samples")
    print(f"  Period length     : {len(period)} samples")
    print(f"  Buffer length     : {len(buffer)} samples")
    print(f"  Burst power (avg) : {np.mean(np.abs(burst)**2):.4f}")
    print(f"  Peak amplitude    : {np.max(np.abs(burst)):.4f}")
    print(f"  Peak-to-Avg ratio : {np.max(np.abs(burst)**2) / np.mean(np.abs(burst)**2):.2f}")
    print("=" * 45)

    # --- Plot 1: Time domain burst ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    axes[0].plot(np.abs(burst))
    axes[0].axvline(x=63, color='r', linestyle='--', label='End of ZC preamble')
    axes[0].set_title("Burst Envelope |burst[n]|")
    axes[0].set_xlabel("Sample index")
    axes[0].set_ylabel("Magnitude")
    axes[0].legend()

    # --- Plot 2: One full period (burst + guard zeros) ---
    axes[1].plot(np.abs(period))
    axes[1].axvline(x=SAMPLES_PER_BURST, color='g', linestyle='--', label='Start of guard')
    axes[1].set_title("One TX Period (burst + guard zeros)")
    axes[1].set_xlabel("Sample index")
    axes[1].set_ylabel("Magnitude")
    axes[1].legend()

    # --- Plot 3: Frequency domain of one OFDM symbol ---
    one_symbol_td = burst[CP_LEN + ZC_LENGTH : ZC_LENGTH + 2 * (FFT_SIZE + CP_LEN)]
    one_symbol_fd = np.fft.fftshift(np.fft.fft(one_symbol_td[:FFT_SIZE]))
    axes[2].stem(np.abs(one_symbol_fd), markerfmt='C0o', linefmt='C0-', basefmt='k-')
    axes[2].set_title("Frequency Domain — First OFDM Symbol (should show 52 equal pilots)")
    axes[2].set_xlabel("Subcarrier index (centered)")
    axes[2].set_ylabel("Magnitude")

    plt.tight_layout()
    plt.savefig("tx_verification.png")
    print("\nPlot saved to tx_verification.png")