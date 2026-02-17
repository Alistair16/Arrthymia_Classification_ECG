import numpy as np
import matplotlib.pyplot as plt
import math

def quick_plot(x, lead=0, n=1000):
    plt.plot(x[lead, :n])
    plt.title(f"Lead {lead}")
    plt.show()


def plot_ecg_stacked(arr, fs=500, n_signals=5, lead_names=None,
                     spacing_factor=2, linewidth=1):
    """
    Plot stacked ECG-style signals.

    Parameters
    ----------
    arr : ndarray, shape (n_leads, n_samples)
        Signal array.
    fs : float, optional
        Sampling frequency (Hz).
    n_signals : int, optional
        Number of rows to plot.
    lead_names : list of str, optional
        Names of signals/leads.
    spacing_factor : float, optional
        Controls vertical spacing between signals.
    linewidth : float, optional
        Line width of traces.
    """

    n_signals = min(n_signals, arr.shape[0])
    time = np.arange(arr.shape[1]) / fs

    # Automatic spacing
    spacing = np.max(np.abs(arr[:n_signals])) * spacing_factor

    plt.figure(figsize=(12, 7))

    for i in range(n_signals):
        plt.plot(time, arr[i] + i * spacing, linewidth=linewidth)

    # Lead labels
    if lead_names is None:
        lead_names = [f"Lead {i+1}" for i in range(n_signals)]

    plt.yticks([i * spacing for i in range(n_signals)],
               lead_names[:n_signals])

    plt.xlabel("Time (s)")
    plt.title("Stacked ECG Signals")
    plt.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()
    plt.show()
