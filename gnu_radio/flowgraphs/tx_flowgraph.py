import numpy as np
import adi
import sys
import os
import time

sys.path.append(os.path.join(os.path.dirname(__file__), '../../python/utils'))
from ofdm_params import (
    CENTER_FREQ, SAMPLE_RATE, TX_GAIN,
    SAMPLES_PER_PERIOD, BURST_RATE
)
from ofdm_transmitter import generate_tx_period


# =============================================================================
# Pluto TX Configuration
# =============================================================================

def configure_pluto_tx(sdr: adi.Pluto) -> None:
    sdr.sample_rate           = int(SAMPLE_RATE)
    sdr.tx_rf_bandwidth       = int(SAMPLE_RATE)
    sdr.tx_lo                 = int(CENTER_FREQ)
    sdr.tx_hardwaregain_chan0 = TX_GAIN
    sdr.tx_cyclic_buffer      = False   # disable cyclic — push periods manually

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

def run_tx(duration_sec: float = 30.0) -> None:

    # 1. Connect
    print("\nConnecting to Pluto...")
    sdr = adi.Pluto('ip:192.168.2.1')

    # 2. Configure
    configure_pluto_tx(sdr)

    # 3. Generate exactly 50000 sample period
    print(f"\nGenerating TX period (50000 samples)...")
    tx_period = generate_tx_period()
    print(f"  Period length    : {len(tx_period)} samples")
    print(f"  Period duration  : {len(tx_period)/SAMPLE_RATE*1000:.1f} ms")
    print(f"  Expected rate    : ~121 Hz")
    print(f"  Transmitting for : {duration_sec:.1f} seconds\n")

    # 4. Push in loop — one period per push
    print("Starting transmission... (Ctrl+C to stop early)")
    start       = time.time()
    burst_count = 0

    try:
        while time.time() - start < duration_sec:
            sdr.tx(tx_period)
            burst_count += 1
            elapsed   = time.time() - start
            remaining = duration_sec - elapsed
            print(f"\r  Bursts sent: {burst_count}  "
                  f"Elapsed: {elapsed:.1f}s  "
                  f"Remaining: {remaining:.1f}s   ",
                  end='', flush=True)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")

    # 5. Done
    elapsed = time.time() - start
    print(f"\n\nDone. Sent {burst_count} bursts in {elapsed:.1f}s")
    print(f"Effective burst rate: {burst_count/elapsed:.1f} Hz")

# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OFDM Channel Sounder TX")
    parser.add_argument('--duration', type=float, default=30.0,
                        help='Transmission duration in seconds (default: 30)')
    args = parser.parse_args()

    run_tx(duration_sec=args.duration)