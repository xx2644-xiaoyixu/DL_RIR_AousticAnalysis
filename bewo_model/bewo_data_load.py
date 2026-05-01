import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

def audio_to_logmel(left_path, right_path, sr=44100, n_fft=1024, hop_length=256, n_mels=128):
    """
    Member 3 Task: Preprocess audio files to images (Log-Mel Spectrograms).
    """
    left, _ = librosa.load(left_path, sr=sr, mono=True)
    right, _ = librosa.load(right_path, sr=sr, mono=True)

    min_len = min(len(left), len(right))
    left, right = left[:min_len], right[:min_len]

    left_mel = librosa.feature.melspectrogram(y=left, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)
    right_mel = librosa.feature.melspectrogram(y=right, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, power=2.0)

    left_logmel = librosa.power_to_db(left_mel, ref=1.0)
    right_logmel = librosa.power_to_db(right_mel, ref=1.0)

    # Output shape: (2, n_mels, Time) which matches PyTorch (Channels, Freq, Time)
    img_feature = np.stack([left_logmel, right_logmel], axis=0)
    return img_feature.astype(np.float32)

def extract_and_save_dataset(label_csv, split_name, audio_dir, output_dir):
    """
    Extracts features natively for the BEWO model, avoiding dependencies on Member 2's pre-extracted arrays.
    """
    df = pd.read_csv(label_csv)
    # Filter by split if needed, though usually label_csv is already split-specific
    if 'split' in df.columns:
        df = df[df['split'] == split_name]

    os.makedirs(output_dir, exist_ok=True)
    out_x_path = os.path.join(output_dir, f"{split_name}_X_logmel.npy")
    out_y_path = os.path.join(output_dir, f"{split_name}_Y_labels.npy")

    if os.path.exists(out_x_path) and os.path.exists(out_y_path):
        print(f"[{split_name}] BEWO pre-extracted features already exist at {output_dir}. Skipping extraction.")
        return out_x_path, out_y_path

    print(f"[{split_name}] BEWO: Extracting audio to image...")
    X, Y = [], []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        left_path = os.path.join(audio_dir, split_name, row["left_file"])
        right_path = os.path.join(audio_dir, split_name, row["right_file"])
        
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            continue
            
        feature = audio_to_logmel(left_path, right_path)
        X.append(feature)
        
        # Capture the targets
        Y.append([int(row["DRR_class"]), int(row["C80_class"]), int(row["RT60_class"])])

    X_np = np.stack(X, axis=0)
    Y_np = np.array(Y, dtype=np.int64)

    np.save(out_x_path, X_np)
    np.save(out_y_path, Y_np)
    print(f"[{split_name}] Saved BEWO features to {out_x_path}")
    return out_x_path, out_y_path

def load_bewo_arrays(x_path, y_path):
    x = np.load(x_path).astype(np.float32)
    y = np.load(y_path).astype(np.int64)
    # If using the BEWO native extraction, shape is already (N, 2, F, T)
    # But if somehow it's (N, F, T, 2) from an older extraction, transpose it.
    if x.ndim == 4 and x.shape[-1] == 2:
        x = np.transpose(x, (0, 3, 1, 2))
    return x, y

def make_pytorch_loader(logmels, labels, batch_size=16, shuffle=True):
    x_tensor = torch.tensor(logmels, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)
    dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
