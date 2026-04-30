# Data generator for training
import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import TensorDataset, DataLoader

# set the path of dataset and labels
FEATURE_DIR = r"C:\data\ir_features"
LABEL_CSV = r"C:\data\ir_labels.csv"

def waveform_to_logmel(
    x,
    sr=44100,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=128,
):
    """
    Input:
        x: shape (2, samples)

    Output:
        feature: shape (n_mels, T, 2)
    """

    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"Expected shape (2, samples), got {x.shape}")

    left = x[0].astype(np.float32)
    right = x[1].astype(np.float32)

    left_mel = librosa.feature.melspectrogram(
        y=left,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )

    right_mel = librosa.feature.melspectrogram(
        y=right,
        sr=sr,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        power=2.0,
    )

    left_logmel = librosa.power_to_db(left_mel, ref=1.0)
    right_logmel = librosa.power_to_db(right_mel, ref=1.0)

    feature = np.stack([left_logmel, right_logmel], axis=-1)
    # feature: (n_mels, T, 2)

    return feature.astype(np.float32)

def load_one_sample(filename, feature_dir=FEATURE_DIR):
    path = os.path.join(feature_dir, filename)

    x = np.load(path).astype(np.float32)
    # x: (2, samples)

    if x.ndim != 2 or x.shape[0] != 2:
        raise ValueError(f"{filename} should have shape (2, samples), but got {x.shape}")

    feature = waveform_to_logmel(x)
    # feature: (n_mels, T, 2)

    return feature

def load_dataset(FEATURE_DIR = r"C:\data\ir_features", LABEL_CSV = r"C:\data\ir_labels.csv"):
    df = pd.read_csv(LABEL_CSV)

    X = []
    Y = []

    for _, row in df.iterrows():
        filename = row["filename"]

        x = load_one_sample(filename, FEATURE_DIR)

        y = np.array(
            [row["DRR"], row["C80"], row["RT60"]],
            dtype=np.float32,
        )

        X.append(x)
        Y.append(y)

    # logmels shape = (N, n_mels, T, 2)
    logmels = np.stack(X, axis=0)
    # labels shape = (N, 3)
    labels = np.stack(Y, axis=0)

    np.save("X_logmel.npy", logmels)
    np.save("Y_labels.npy", labels)

    return logmels, labels

def make_pytorch_loader(logmels, labels, batch_size=16, shuffle=True):
    """
    logmels: (N, F, T, 2)
    labels:  (N, 3)
    """

    logmels_t = np.transpose(logmels, (0, 3, 1, 2))
    # (N, F, T, 2) -> (N, 2, F, T)

    x_tensor = torch.tensor(logmels_t, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader


