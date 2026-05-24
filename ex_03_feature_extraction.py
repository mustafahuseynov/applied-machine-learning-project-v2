import numpy as np
import pandas as pd
from scipy.signal import detrend, windows


def find_dominant_frequencies(x: np.ndarray, fs: int) -> np.ndarray:
    """
    Calculates the dominant frequencies of multiple input signals with the fast fourier transformation.

    Args:
        x (np.ndarray): The input signals, shape: (num_samples, seq_len).
        fs (int): The sampling frequency of the signals.

    Returns:
        np.ndarray: The dominant frequencies for each signal, shape: (num_samples,).
    """
    x_detrended = detrend(x, axis=1)
    window = windows.hann(x.shape[1])
    x_windowed = x_detrended * window

    fft = np.fft.rfft(x_windowed, axis=1)
    freqs = np.fft.rfftfreq(x.shape[1], d=1 / fs)
    magnitude = np.abs(fft)

    return freqs[np.argmax(magnitude, axis=1)]


def extract_features(data: np.ndarray, labels: np.ndarray) -> pd.DataFrame:
    """
    Extract 20 different features from the data.
    Args:
        data (np.ndarray): The data to extract features from.
        labels (np.ndarray): The labels of the data.
    Returns:
        pd.DataFrame: The extracted features.
    """
    current = data[:, :, 0]   # (1000, 200)
    voltage = data[:, :, 1]   # (1000, 200)

    def compute_basic_stats(signal: np.ndarray, prefix: str) -> pd.DataFrame:
        mean = np.mean(signal, axis=1)          # (1000,)
        std  = np.std(signal, axis=1)           # (1000,)
        std_safe = np.where(std == 0, 1e-10, std)

        diff_from_mean = signal - mean[:, np.newaxis]

        return pd.DataFrame({
            f"{prefix}_mean":     mean,
            f"{prefix}_std":      std,
            f"{prefix}_min":      np.min(signal, axis=1),
            f"{prefix}_max":      np.max(signal, axis=1),
            f"{prefix}_median":   np.median(signal, axis=1),
            f"{prefix}_rms":      np.sqrt(np.mean(signal**2, axis=1)),
            f"{prefix}_skew":     pd.DataFrame(signal).skew(axis=1),
            f"{prefix}_kurtosis": pd.DataFrame(signal).kurtosis(axis=1),
            f"{prefix}_range":    np.ptp(signal, axis=1),
            f"{prefix}_dom_freq": find_dominant_frequencies(signal, fs=1000),
            f"{prefix}_skewness": np.mean(diff_from_mean**3, axis=1) / std_safe**3,
            f"{prefix}_kurt_manual": (np.mean(diff_from_mean**4, axis=1) / std_safe**4) - 3.0,
        })

    features_current = compute_basic_stats(current, "current")
    features_voltage = compute_basic_stats(voltage, "voltage")

    features = pd.concat([features_current, features_voltage], axis=1)
    features["label"] = labels

    features.head().to_csv("features.csv", index=False)

    return features
