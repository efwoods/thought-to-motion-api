import os
import gc
from glob import glob
import joblib
import numpy as np
import pandas as pd
import time

# Scipy
from scipy.signal import butter, filtfilt, iirnotch, hilbert
from scipy.stats import kurtosis
from scipy.io import savemat

# Scikit-Learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# Pytorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import random_split
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


# Noise Filters
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = fs / 2.0
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")
    return b, a


def bandpass_filter(data, lowcut=1.0, highcut=200.0, fs=1000.0, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data, axis=0)


# Apply after bandpass
def notch_filter(data, freq=60.0, fs=1000.0, quality=30.0):
    b, a = iirnotch(freq, quality, fs)
    return filtfilt(b, a, data, axis=0)


# Noise Metrics for evaluation
def compute_rmse(true, estimate):
    return np.sqrt(np.mean((true - estimate) ** 2))


# Kurtosis signal reduction > 0 shows a denoised signal
def proportion_of_positive_kurtosis_signals(kurtosis_raw, kurtosis_denoised):
    return (
        np.array([(kurtosis_raw - kurtosis_denoised) > 0]).sum() / len(kurtosis_raw)
    ) * 100


# Use a Standard scaler to reduce the mean to 0 and std to 1


# Computing the power envelope of each channel


def band_power_envelope(
    ecog_signal: np.ndarray,
    lowcut: float,
    highcut: float,
    fs: float = 1000.0,
    order: int = 4,
) -> np.ndarray:
    """Computes band-limited envelope via Hilbert transform.
    Parameters
    ----------
    self.ecog_signal : np.ndarray (T, channels)
        This is the ecog signal that has been filtered.
    lowcut : float
        This is the lower band limit in Hz.
    highcut : float
        This is the upper band limit in Hz.
    fs : float, optional
        This is the frequency of the sample., by default 1000.0
    order : int, optional
        This is the Butterworth order, by default 4
    Returns
    -------
    np.ndarray
        envelope
    """
    # 1. Narrowband bandpass
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    narrow = filtfilt(b, a, ecog_signal, axis=0)
    # 2. Hilbert transform to get analytic signal
    analytic = hilbert(narrow, axis=0)
    # 3. Envelope = absolute value
    envelope = np.abs(analytic)
    return envelope


def multiband_features(ecog_raw: np.ndarray, fs: float = 1000.0) -> np.ndarray:
    """Builds concatenated band-power features for μ, β, and high-gamma using a Hilbert transform.
    Parameters
    ----------
    ecog_raw : np.ndarray
        (T, 64)
    fs : float, optional
        Frequency of the sample, by default 1000.0
    Returns
    -------
    np.ndarray
        features: (T, 64, 3) (μ, β, high-gamma per electrode)
    """
    mu_env = band_power_envelope(ecog_raw, lowcut=8.0, highcut=13.0, fs=fs)
    beta_env = band_power_envelope(ecog_raw, lowcut=13.0, highcut=30.0, fs=fs)
    hg_env = band_power_envelope(ecog_raw, lowcut=70.0, highcut=200.0, fs=fs)
    # Concatenate along channel dimension
    return np.concatenate([mu_env, beta_env, hg_env], axis=1)


def create_overlapping_windows(
    ecog_values: np.ndarray,
    motion_values: np.ndarray,
    window_size: int = 20,
    hop_size: int = 10,
):
    """Builds overlapping windows to increase sample count and capture smoother transitions.

    Parameters
    ----------
    ecog_values : np.ndarray
        (T, features)
    motion_values : np.ndarray
        (T_motion, 3)_
    window_size : int, optional
        number of timepoints per window, by default 20
    hop_size : int, optional
        step bewteen windows, by default 10
    """
    num_samples, num_features = ecog_values.shape
    print(f"number of Samples")
    max_windows = (num_samples - window_size) // hop_size + 1
    X_list = []
    y_list = []
    for w in range(max_windows):
        start = w * hop_size
        end = start + window_size
        if end > num_samples:
            break
        # Assign label as motion at center of window (or last timepoint)
        X_list.append(ecog_values[start:end, :])
        y_list.append(motion_values[min(end - 1, motion_values.shape[0] - 1), :])
    X = np.stack(X_list, axis=0)
    y = np.stack(y_list, axis=0)
    return X, y


def predict_and_export(model, data_loader, device, output_file_path):
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    predictions = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    # Save as .mat file for visualization
    savemat(output_file_path, {"predictions": predictions, "targets": targets})
    print("Saved predictions to ecog_predictions.mat")

    return predictions, targets


# Defining Preprocessing for the raw data
class PreprocessData:
    def __init__(self, ecog_file_path, motion_file_path):
        self.ecog_file_path = ecog_file_path
        self.motion_file_path = motion_file_path
        self.ecog_data = None
        self.motion_data = None
        self.filtered_ecog = None
        self.scaled_ecog = None
        self.X = None
        self.y = None
        self.scaler = None

    def process(
        self,
        eval=False,
        window_size=20,
        duration_limit=900,
        function_timer=False,
    ):
        if function_timer:
            preprocessing_start_time = time.perf_counter_ns()
        if function_timer:
            reading_time_start = time.perf_counter_ns()
        self.read_data()
        if function_timer:
            reading_time_stop = time.perf_counter_ns()

        if function_timer:
            common_average_reference_time_start = time.perf_counter_ns()
        self.common_average_reference()
        if function_timer:
            common_average_reference_time_stop = time.perf_counter_ns()

        if function_timer:
            filtering_time_start = time.perf_counter_ns()
        self.filter_signal(eval=eval)
        if function_timer:
            filtering_time_stop = time.perf_counter_ns()

        if function_timer:
            formatting_time_start = time.perf_counter_ns()
        self.format_data(window_size=window_size, duration_limit=duration_limit)
        if function_timer:
            formatting_time_stop = time.perf_counter_ns()
        if function_timer:
            preprocessing_stop_time = time.perf_counter_ns()

        if function_timer:
            print(
                f"Reading Time: {(reading_time_stop - reading_time_start) / 1e9} seconds"
            )
        if function_timer:
            print(
                f"Common Average Reference Time: {(common_average_reference_time_stop - common_average_reference_time_start) / 1e9} seconds"
            )
        if function_timer:
            print(
                f"Filtering Time: {(filtering_time_stop - filtering_time_start) / 1e9} seconds"
            )
        if function_timer:
            print(
                f"Formatting Time: {(formatting_time_stop - formatting_time_start) / 1e9} seconds"
            )
        if function_timer:
            print(
                f"Total_time_spent_preprocessing: {(preprocessing_stop_time - preprocessing_start_time)//1e9}"
            )

        return self.X, self.y

    def process_portion(
        self,
        eval=False,
        window_size=20,
        duration_limit=900,
        function_timer=False,
    ):
        # Prereq: Data was manually read and is assigned after init of preprocessor with self.ecog_data & self.motion_data assignment
        if function_timer:
            preprocessing_start_time = time.perf_counter_ns()
        if function_timer:
            common_average_reference_time_start = time.perf_counter_ns()
        self.common_average_reference()
        if function_timer:
            common_average_reference_time_stop = time.perf_counter_ns()

        if function_timer:
            filtering_time_start = time.perf_counter_ns()
        self.filter_signal(eval=eval)
        if function_timer:
            filtering_time_stop = time.perf_counter_ns()

        if function_timer:
            formatting_time_start = time.perf_counter_ns()
        self.format_data(window_size=window_size, duration_limit=duration_limit)
        if function_timer:
            formatting_time_stop = time.perf_counter_ns()
        if function_timer:
            preprocessing_stop_time = time.perf_counter_ns()

        if function_timer:
            print(
                f"Common Average Reference Time: {(common_average_reference_time_stop - common_average_reference_time_start) / 1e9} seconds"
            )
        if function_timer:
            print(
                f"Filtering Time: {(filtering_time_stop - filtering_time_start) / 1e9} seconds"
            )
        if function_timer:
            print(
                f"Formatting Time: {(formatting_time_stop - formatting_time_start) / 1e9} seconds"
            )
        if function_timer:
            print(
                f"Total_time_spent_preprocessing: {(preprocessing_stop_time - preprocessing_start_time)//1e9}"
            )

        return self.X, self.y

    def read_data(self):
        print("Reading data")
        self.ecog_data = pd.read_csv(self.ecog_file_path)
        self.motion_data = pd.read_csv(self.motion_file_path)
        print(f"self.ecog_data.shape:{self.ecog_data.shape}")
        print(f"self.motion_data.shape:{self.motion_data.shape}")
        return self

    def common_average_reference(self):
        # Subtract the common mean from the signals
        print(
            "Subtracting common mean from the signals to create common average reference"
        )
        print(
            f"Before Common average reference: self.ecog_data.shape: {self.ecog_data.shape}"
        )
        print(
            f"Before Common average reference: self.motion_data.shape: {self.motion_data.shape}"
        )
        common_average_reference = np.mean(
            self.ecog_data.drop(["Time", "Fs"], axis=1).values, axis=1, keepdims=1
        )
        ecog_data_values = self.ecog_data[self.ecog_data.columns[1:-1]].values
        ecog_data_common_mean_subtracted = ecog_data_values - common_average_reference
        self.ecog_data[self.ecog_data.columns[1:-1]] = ecog_data_common_mean_subtracted
        print(
            f"After Common average reference: self.ecog_data.shape: {self.ecog_data.shape}"
        )
        print(
            f"After Common average reference: self.motion_data.shape: {self.motion_data.shape}"
        )
        del ecog_data_values, ecog_data_common_mean_subtracted, common_average_reference
        gc.collect()
        return self

    def filter_signal(self, eval=False):
        ecog_raw = self.ecog_data[self.ecog_data.columns[1:-1]].values
        print(f"Raw Data Shape: {ecog_raw.shape}")

        # Apply filters
        print(f"Applying a bandpass filter from 1 KHz to 200 KHz")
        filtered = bandpass_filter(
            ecog_raw, lowcut=1.0, highcut=200.0, fs=1000.0, order=4
        )
        print(f"Removing 60 Hz Electrical Noise with a Notch Filter")
        denoised = notch_filter(filtered, freq=60, fs=1000.0)
        print(f"Denoised Shape: {denoised.shape}")
        # Evaluate filters
        if eval:
            kurt_raw = kurtosis(ecog_raw, axis=0, fisher=True)
            kurt_denoised = kurtosis(denoised, axis=0, fisher=True)
            proportion_of_positive_kurtosis_signals(kurt_raw, kurt_denoised)
            compute_rmse(ecog_raw, denoised)

        # Compute Power Envelopes
        print(
            "Computing Power Envelopes: Builds concatenated band-power features for μ, β, and high-gamma using a Hilbert transform"
        )
        features = multiband_features(denoised, fs=1000.0)  # shape (T, 192)
        print(f"Features Shape of the Multiband Features: {features.shape}")

        # Identify the principal components of the network
        print(f"Identifying 64 Principal components of the network")
        pca = PCA(n_components=64, random_state=42)
        reduced = pca.fit_transform(features)
        print(f"Reduced Shape from PCA: {reduced.shape}")
        # Scale
        print(f"Scaling the data to have a mean of 0 and standard deviation of 1")
        self.scaler = StandardScaler()
        self.scaled_ecog = self.scaler.fit_transform(reduced)

        # Replace in DataFrame
        self.ecog_data = self.ecog_data.copy()
        self.ecog_data[self.ecog_data.columns[1:-1]] = self.scaled_ecog

        # Clean memory
        del ecog_raw, filtered, denoised
        gc.collect()
        return self

    def format_data(self, window_size=20, duration_limit=900):
        print(f"This data has been preprocessed.")
        print(f"Truncating the data to have the same 15 minute limit.")
        ecog_df = self.ecog_data[self.ecog_data["Time"] <= duration_limit]
        motion_df = self.motion_data[self.motion_data["Motion_time"] <= duration_limit]

        ecog_values = ecog_df.drop(columns=["Fs", "Time"]).values
        motion_values = motion_df.drop(columns=["Fsm", "Motion_time"]).values

        print(f"motion_values.shape: {motion_values.shape}")
        print(f"ecog_values.shape: {ecog_values.shape}")

        # Smooth the signal
        print(f"Creating Overlapping Windows of data to Smooth the Signal")
        X, y = create_overlapping_windows(
            ecog_values, motion_values, window_size=20, hop_size=10
        )
        print(f"y.shape: {y.shape}")
        self.X, self.y = X, y

        print(self.X.shape)
        print(self.y.shape)

        # Clean up
        del ecog_values, motion_values
        gc.collect()

    def save(self):
        output_file_path_base = self.ecog_file_path.strip("ecog_data.csv")
        joblib.dump(self.scaler, output_file_path_base + "scaler_ecog.pkl")
        np.save(output_file_path_base + "X.npy", self.X)
        np.save(output_file_path_base + "y.npy", self.y)
