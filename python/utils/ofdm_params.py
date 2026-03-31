import numpy as np


# =============================================================================
# RF Parameters
# =============================================================================
CENTER_FREQ     = 2.4e9       # Hz  - carrier frequency
SAMPLE_RATE     = 20e6        # Hz  - baseband sample rate (= bandwidth)
TX_GAIN         = -10         # dB  - Pluto TX attenuation (0 = max power, -89 = min)
RX_GAIN         = 30          # dB  - Pluto RX gain

# =============================================================================
# OFDM Parameters
# =============================================================================
FFT_SIZE        = 64          # total subcarriers (matches 802.11 structure)
CP_LEN          = 16          # cyclic prefix length in samples (FFT_SIZE / 4)
NUM_PILOTS      = 52          # used subcarriers (center 52 of 64, like 802.11g)
NUM_SYMBOLS     = 8           # OFDM symbols per burst (tune this for burst duration)

# Subcarrier index mapping (zero-frequency centered, like 802.11)
# Skips DC (index 0) and the outer guard bands
PILOT_INDICES   = np.array([
    -26, -25, -24, -23, -22, -21, -20, -19, -18, -17, -16, -15, -14,
    -13, -12, -11, -10,  -9,  -8,  -7,  -6,  -5,  -4,  -3,  -2,  -1,
      1,   2,   3,   4,   5,   6,   7,   8,   9,  10,  11,  12,  13,
     14,  15,  16,  17,  18,  19,  20,  21,  22,  23,  24,  25,  26
])  # shape: (52,)

# Map to FFT bin indices (0-indexed, numpy fft convention)
PILOT_BINS      = np.where(
    PILOT_INDICES < 0,
    PILOT_INDICES + FFT_SIZE,
    PILOT_INDICES
)  # negative freqs wrap to upper half of FFT

# =============================================================================
# Pilot Symbol Values
# =============================================================================
# All pilots set to BPSK +1 — receiver knows this exactly, so H[k] = Y[k] / X[k]
# where X[k] = 1+0j for all k, simplifying estimation to H[k] = Y[k]
PILOT_SYMBOL    = 1.0 + 0.0j
PILOT_VALUES    = np.ones(NUM_PILOTS, dtype=np.complex64) * PILOT_SYMBOL

# =============================================================================
# Zadoff-Chu Preamble Parameters
# =============================================================================
ZC_LENGTH       = 63          # odd prime, fits cleanly before OFDM symbols
ZC_ROOT         = 1           # root index, coprime to ZC_LENGTH

# =============================================================================
# Burst Timing
# =============================================================================
BURST_RATE      = 100         # Hz - how often a full burst is transmitted

# Samples per OFDM symbol (including CP)
SAMPLES_PER_SYMBOL  = FFT_SIZE + CP_LEN               # = 80 samples

# Samples per full burst (ZC preamble + all OFDM symbols)
SAMPLES_PER_BURST   = ZC_LENGTH + (SAMPLES_PER_SYMBOL * NUM_SYMBOLS)

# Inter-burst zero padding to hit exactly 100 Hz at 20 MHz sample rate
SAMPLES_PER_PERIOD  = int(SAMPLE_RATE / BURST_RATE)   # = 200,000 samples
GUARD_SAMPLES       = SAMPLES_PER_PERIOD - SAMPLES_PER_BURST

# =============================================================================
# Sanity Checks (run on import)
# =============================================================================
assert CP_LEN == FFT_SIZE // 4, \
    "CP length should be FFT_SIZE / 4"

assert GUARD_SAMPLES > 0, \
    f"Burst is longer than one period! Reduce NUM_SYMBOLS or increase SAMPLE_RATE.\n" \
    f"SAMPLES_PER_BURST={SAMPLES_PER_BURST}, SAMPLES_PER_PERIOD={SAMPLES_PER_PERIOD}"

assert len(PILOT_INDICES) == NUM_PILOTS, \
    "PILOT_INDICES length does not match NUM_PILOTS"


# =============================================================================
# Summary (printed when run directly)
# =============================================================================
if __name__ == "__main__":
    burst_duration_ms   = SAMPLES_PER_BURST  / SAMPLE_RATE * 1000
    period_duration_ms  = SAMPLES_PER_PERIOD / SAMPLE_RATE * 1000
    guard_duration_ms   = GUARD_SAMPLES      / SAMPLE_RATE * 1000
    duty_cycle          = SAMPLES_PER_BURST  / SAMPLES_PER_PERIOD * 100

    print("=" * 45)
    print("        OFDM System Parameters")
    print("=" * 45)
    print(f"  Center frequency   : {CENTER_FREQ/1e9:.2f} GHz")
    print(f"  Sample rate        : {SAMPLE_RATE/1e6:.1f} MHz")
    print(f"  FFT size           : {FFT_SIZE}")
    print(f"  Cyclic prefix      : {CP_LEN} samples")
    print(f"  Used subcarriers   : {NUM_PILOTS}")
    print(f"  OFDM symbols/burst : {NUM_SYMBOLS}")
    print("-" * 45)
    print(f"  ZC preamble length : {ZC_LENGTH} samples")
    print(f"  Samples per symbol : {SAMPLES_PER_SYMBOL}")
    print(f"  Samples per burst  : {SAMPLES_PER_BURST}")
    print(f"  Guard samples      : {GUARD_SAMPLES}")
    print(f"  Samples per period : {SAMPLES_PER_PERIOD}")
    print("-" * 45)
    print(f"  Burst duration     : {burst_duration_ms:.3f} ms")
    print(f"  Guard duration     : {guard_duration_ms:.3f} ms")
    print(f"  Period duration    : {period_duration_ms:.3f} ms")
    print(f"  Burst rate         : {BURST_RATE} Hz")
    print(f"  TX duty cycle      : {duty_cycle:.2f}%")
    print("=" * 45)