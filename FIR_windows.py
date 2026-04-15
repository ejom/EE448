import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, freqz, windows


def design_fir_filters(
    num_taps: int = 51,
    cutoff: float = 0.3,
    kaiser_beta: float = 8.0,
):
    """
    Design low-pass FIR filters using different window techniques.

    Parameters
    ----------
    num_taps : int
        Number of FIR coefficients (filter length).
    cutoff : float
        Normalized cutoff frequency where 1.0 corresponds to Nyquist.
        Must satisfy 0 < cutoff < 1.
    kaiser_beta : float
        Beta parameter for the Kaiser window.

    Returns
    -------
    dict
        Dictionary mapping window name to filter coefficients.
    """
    filters = {
        "Rectangular": firwin(num_taps, cutoff, window="boxcar"),
        "Hann": firwin(num_taps, cutoff, window="hann"),
        "Hamming": firwin(num_taps, cutoff, window="hamming"),
        "Kaiser": firwin(num_taps, cutoff, window=("kaiser", kaiser_beta)),
    }
    return filters


def generate_windows(num_taps: int = 51, kaiser_beta: float = 8.0):
    """
    Generate the raw windows themselves for plotting.
    """
    win_dict = {
        "Rectangular": windows.boxcar(num_taps),
        "Hann": windows.hann(num_taps),
        "Hamming": windows.hamming(num_taps),
        "Kaiser": windows.kaiser(num_taps, beta=kaiser_beta),
    }
    return win_dict


def plot_windows(win_dict: dict):
    """
    Plot the time-domain window shapes.
    """
    plt.figure(figsize=(10, 6))
    for name, w in win_dict.items():
        plt.plot(w, label=name)
    plt.title("Window Functions")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_impulse_responses(filters: dict):
    """
    Plot FIR filter coefficients (impulse responses).
    """
    plt.figure(figsize=(10, 6))
    for name, h in filters.items():
        plt.stem(
            np.arange(len(h)),
            h,
            linefmt="-",
            markerfmt="o",
            basefmt=" ",
            label=name,
        )
    plt.title("FIR Filter Impulse Responses")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def plot_frequency_responses(filters: dict, worN: int = 4096):
    """
    Plot the magnitude frequency response of each FIR filter in dB.
    """
    plt.figure(figsize=(10, 6))
    for name, h in filters.items():
        w, H = freqz(h, worN=worN)
        freq = w / np.pi  # normalized frequency: 1 = Nyquist
        magnitude_db = 20 * np.log10(np.maximum(np.abs(H), 1e-10))
        plt.plot(freq, magnitude_db, label=name)

    plt.title("Magnitude Frequency Responses")
    plt.xlabel("Normalized Frequency (×π rad/sample)")
    plt.ylabel("Magnitude (dB)")
    plt.ylim([-120, 5])
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def main():
    num_taps = 51
    cutoff = 0.3
    kaiser_beta = 8.0

    windows_dict = generate_windows(num_taps=num_taps, kaiser_beta=kaiser_beta)
    filters = design_fir_filters(
        num_taps=num_taps,
        cutoff=cutoff,
        kaiser_beta=kaiser_beta,
    )

    plot_windows(windows_dict)
    plot_impulse_responses(filters)
    plot_frequency_responses(filters)

    plt.show()


if __name__ == "__main__":
    main()