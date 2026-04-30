# Data generator for training
import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import TensorDataset, DataLoader

# define the path of data and label
DATASET_ROOT = r"C:\Users\14362\Desktop\DL\Assignment\Group Project\DL_RIR_AousticAnalysis\dataset"
AUDIO_DIR = os.path.join(DATASET_ROOT, "audio")
LABEL_DIR = os.path.join(DATASET_ROOT, "labels")

TRAIN_CSV = os.path.join(LABEL_DIR, "train_labels_classification.csv")
VAL_CSV = os.path.join(LABEL_DIR, "validation_labels_classification.csv")
TEST_CSV = os.path.join(LABEL_DIR, "test_labels_classification.csv")


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
    return feature.astype(np.float32)


def load_one_sample(row, audio_dir=AUDIO_DIR, sr=44100):
    split = row["split"]
    class_name = f"class_{int(row['DRR_class'])}"

    left_path = os.path.join(audio_dir, split, class_name, row["left_file"])
    right_path = os.path.join(audio_dir, split, class_name, row["right_file"])

    if not os.path.exists(left_path):
        raise FileNotFoundError(left_path)
    if not os.path.exists(right_path):
        raise FileNotFoundError(right_path)

    left, _ = librosa.load(left_path, sr=sr, mono=True)
    right, _ = librosa.load(right_path, sr=sr, mono=True)

    min_len = min(len(left), len(right))
    left = left[:min_len]
    right = right[:min_len]

    x = np.stack([left, right], axis=0)
    feature = waveform_to_logmel(x, sr=sr)

    return feature


def load_dataset(label_csv, audio_dir=AUDIO_DIR, save_prefix=None):
    df = pd.read_csv(label_csv)

    X = []
    Y = []

    for _, row in df.iterrows():
        x = load_one_sample(row, audio_dir)

        y = np.array(
            [
                row["DRR_class"],
                row["RT60_class"],
                row["C80_class"],
            ],
            dtype=np.int64,
        )

        X.append(x)
        Y.append(y)

    logmels = np.stack(X, axis=0)
    labels = np.stack(Y, axis=0)

    if save_prefix is not None:
        np.save(f"{save_prefix}_X_logmel.npy", logmels)
        np.save(f"{save_prefix}_Y_labels.npy", labels)

    return logmels, labels


def make_pytorch_loader(logmels, labels, batch_size=16, shuffle=True):
    """
    logmels: (N, F, T, 2)
    labels:  (N, 3), classification labels:
             [DRR_class, RT60_class, C80_class]
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


def load_train_val_test(batch_size=16):
    train_X, train_Y = load_dataset(TRAIN_CSV, save_prefix="train")
    val_X, val_Y = load_dataset(VAL_CSV, save_prefix="val")
    test_X, test_Y = load_dataset(TEST_CSV, save_prefix="test")

    train_loader = make_pytorch_loader(train_X, train_Y, batch_size=batch_size, shuffle=True)
    val_loader = make_pytorch_loader(val_X, val_Y, batch_size=batch_size, shuffle=False)
    test_loader = make_pytorch_loader(test_X, test_Y, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader


