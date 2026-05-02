# Data generator for training
import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import TensorDataset, DataLoader


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
    return feature.astype(np.float32)


def load_one_sample(row, audio_dir, sr=44100):
    left_path = os.path.join(audio_dir, row["left_file"])
    right_path = os.path.join(audio_dir, row["right_file"])

    left, _ = librosa.load(left_path, sr=sr, mono=True)
    right, _ = librosa.load(right_path, sr=sr, mono=True)

    min_len = min(len(left), len(right))
    x = np.stack([left[:min_len], right[:min_len]], axis=0)

    return waveform_to_logmel(x, sr=sr)

# define the path that saves the logmels and labels for each group of the dataset
SAVE_DIR = r"C:\Users\14362\Desktop\DL\Assignment\Group Project\DL_RIR_AousticAnalysis\baseline_model\bl_dataset_npy"

LABEL_COLUMNS = ["DRR_class", "C80_class", "RT60_class", "ILD_class", "ITD_class"]

def load_data(audio_dir, label_csv, save_prefix=None):
    df = pd.read_csv(label_csv)

    X = []
    Y = []

    for _, row in df.iterrows():
        x = load_one_sample(row, audio_dir)

        y = row[LABEL_COLUMNS].to_numpy(dtype=np.int64)

        X.append(x)
        Y.append(y)

    logmels = np.stack(X, axis=0)
    labels = np.stack(Y, axis=0)

    if save_prefix is not None:
        os.makedirs(SAVE_DIR, exist_ok=True)
        x_save_path = os.path.join(SAVE_DIR, f"{save_prefix}_X_logmel.npy")
        y_save_path = os.path.join(SAVE_DIR, f"{save_prefix}_Y_labels.npy")
        np.save(x_save_path, logmels)
        np.save(y_save_path, labels)

    return logmels, labels


def make_pytorch_loader(logmels, labels, batch_size=16, shuffle=True):
    """
    logmels: (N, F, T, 2)
    labels:  (N, 5), classification labels:
             [DRR_class, C80_class, RT60_class, ILD_class, ITD_class]
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

# example of loading data
def load_train_val_test(batch_size=16):
    train_X, train_Y = load_data(TRAIN_DIR, TRAIN_CSV, save_prefix="train")
    val_X, val_Y = load_data(VAL_DIR, VAL_CSV, save_prefix="val")
    test_X, test_Y = load_data(TEST_DIR, TEST_CSV, save_prefix="test")

    train_loader = make_pytorch_loader(train_X, train_Y, batch_size=batch_size, shuffle=True)
    val_loader = make_pytorch_loader(val_X, val_Y, batch_size=batch_size, shuffle=False)
    test_loader = make_pytorch_loader(test_X, test_Y, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


