import numpy as np
import adi
import sys
import os
import time

# Import our utilities
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python/utils'))
from ofdm_params import (
    CENTER_FREQ, SAMPLE_RATE, TX_GAIN,
    SAMPLES_PER_PERIOD, BURST_RATE
)
from ofdm_transmitter import generate_tx_buffer


# =============================================================================
# Pluto TX Configuration
# =============================================================================

def configure_pluto_tx(sdr: adi.Pluto) -> None:
    """
    Configure the Pluto SDR for transmission.
    """
    sdr.sample_rate          = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth      = int(SAMPLE_RATE)
    sdr.tx_lo                = int(CENTER_FREQ)
    sdr.tx_hardwaregain_chan0 = TX_GAIN        # negative = attenuation on Pluto
    sdr.tx_cyclic_buffer     = True            # hardware repeats buffer automatically

    print("=" * 45)
    print("        Pluto TX Configuration")
    print("=" * 45)
    print(f"  Sample rate      : {sdr.sample_rate/1e6:.1f} MHz")
    print(f"  Center frequency : {sdr.tx_lo/1e9:.4f} GHz")
    print(f"  TX gain          : {sdr.tx_hardwaregain_chan0} dB")
    print(f"  Cyclic buffer    : {sdr.tx_cyclic_buffer}")
    print("=" * 45)


# =============================================================================
# Main TX Loop
# =============================================================================

def run_tx(duration_sec: float = 30.0, num_periods: int = 10) -> None:
    """
    Transmit OFDM bursts continuously for a given duration.

    Parameters
    ----------
    duration_sec : float
        How long to transmit in seconds (default 30s)
    num_periods : int
        Number of periods in the TX buffer loaded to the Pluto (default 10)
        With tx_cyclic_buffer=True the hardware loops this automatically.
    """

    # 1. Connect to Pluto
    print("\nConnecting to Pluto...")
    sdr = adi.Pluto('ip:192.168.2.1')

    # 2. Configure TX
    configure_pluto_tx(sdr)

    # 3. Generate TX buffer
    print(f"\nGenerating TX buffer ({num_periods} periods)...")
    tx_buffer = generate_tx_buffer(num_periods=num_periods)
    print(f"  Buffer length    : {len(tx_buffer)} samples")
    print(f"  Buffer duration  : {len(tx_buffer)/SAMPLE_RATE*1000:.1f} ms")
    print(f"  Will cycle for   : {duration_sec:.1f} seconds")

    # 4. Load buffer to Pluto and start transmitting
    #    With tx_cyclic_buffer=True, the Pluto hardware loops this buffer
    #    indefinitely at the hardware level — no Python loop needed
    print("\nStarting transmission...")
    sdr.tx(tx_buffer)
    print(f"Transmitting at {CENTER_FREQ/1e9:.3f} GHz — {BURST_RATE} bursts/sec")
    print("Press Ctrl+C to stop early.\n")

    # 5. Wait for duration then stop
    try:
        start = time.time()
        while time.time() - start < duration_sec:
            elapsed = time.time() - start
            remaining = duration_sec - elapsed
            print(f"\r  Elapsed: {elapsed:.1f}s / {duration_sec:.1f}s  "
                  f"Remaining: {remaining:.1f}s", end='', flush=True)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # 6. Clean shutdown
    print("\nStopping transmission...")
    sdr.tx_destroy_buffer()
    print("TX buffer destroyed. Done.")


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OFDM Channel Sounder TX")
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Transmission duration in seconds (default: 30)')
    parser.add_argument('--periods',  type=int,   default=10,
                        help='Number of periods in TX buffer (default: 10)')
    args = parser.parse_args()

    run_tx(duration_sec=args.duration, num_periods=args.periods)