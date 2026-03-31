import numpy as np
import matplotlib.pyplot as plt


def generate_zc(N: int = 63, u: int = 1) -> np.ndarray:
    """
    Generate a Zadoff-Chu sequence.

    Parameters
    ----------
    N : int
        Sequence length. Should be an odd prime (default 63).
    u : int
        Root index. Must be coprime to N (default 1).

    Returns
    -------
    np.ndarray
        Complex ZC sequence of length N.
    """
    n = np.arange(N)
    return np.exp(-1j * np.pi * u * n * (n + 1) / N)


def verify_autocorrelation(zc: np.ndarray, plot: bool = True) -> None:
    """
    Verify the ZC sequence has a single autocorrelation spike.
    """
    autocorr = np.abs(np.fft.ifft(
        np.fft.fft(zc) * np.conj(np.fft.fft(zc))
    ))

    peak   = np.max(autocorr)
    others = np.sort(autocorr)[-2]  # second highest value

    print(f"ZC length      : {len(zc)}")
    print(f"Autocorr peak  : {peak:.4f}")
    print(f"Second highest : {others:.4f}")
    print(f"Peak-to-side   : {peak / others:.1f}x  (should be >> 1)")

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(autocorr)
        plt.title("Zadoff-Chu Autocorrelation")
        plt.xlabel("Lag (samples)")
        plt.ylabel("Magnitude")
        plt.tight_layout()
        plt.savefig("zc_autocorr.png")
        plt.show()
        print("Plot saved to zc_autocorr.png")


if __name__ == "__main__":
    zc = generate_zc(N=63, u=1)
    verify_autocorrelation(zc, plot=True)
    