import os
import numpy as np
import pandas as pd
import librosa
import torch
from torch.utils.data import TensorDataset, DataLoader

# Physical threshold binning functions
def bin_drr(val):
    """
    Direct-to-Reverberant Ratio (DRR)
    Class 0 (< -5 dB): Reverberation dominant, far from source
    Class 1 (-5 ~ 5 dB): Transition region
    Class 2 (>= 5 dB): Direct sound dominant, close to source
    """
    if val < -5.0: return 0
    elif val < 5.0: return 1
    else: return 2

def bin_c80(val):
    """
    Clarity (C80)
    Class 0 (< 0 dB): Poor clarity
    Class 1 (0 ~ 5 dB): Fair clarity
    Class 2 (>= 5 dB): High clarity
    """
    if val < 0.0: return 0
    elif val < 5.0: return 1
    else: return 2

def bin_rt60(val):
    """
    Reverberation Time (RT60)
    Class 0 (< 0.3s): Short reverb (e.g. studios, small rooms)
    Class 1 (0.3s ~ 1.0s): Medium reverb (e.g. living rooms, classrooms)
    Class 2 (>= 1.0s): Long reverb (e.g. halls, churches)
    """
    if val < 0.3: return 0
    elif val < 1.0: return 1
    else: return 2

def load_one_sample(filename, feature_dir):
    path = os.path.join(feature_dir, filename)
    if not os.path.exists(path):
        # Fallback handle if data directory changes
        return None
        
    x = np.load(path).astype(np.float32)
    
    # The saved features are already extracted binaural mel-spectrograms (2, 32, 64)
    # So we don't need to call waveform_to_logmel here
    if x.ndim != 3 or x.shape[0] != 2:
        raise ValueError(f"{filename} should have shape (2, F, T), but got {x.shape}")

    return x

def load_dataset(feature_dir, label_csv):
    df = pd.read_csv(label_csv)

    X = []
    Y = []

    for _, row in df.iterrows():
        # Fixed the column name from 'filename' to 'feature_file'
        filename = row["feature_file"]
        
        x = load_one_sample(filename, feature_dir)
        if x is None:
            print(f"Warning: File {filename} not found in {feature_dir}. Skipping.")
            continue

        # Discretize continuous values into categorical classes
        drr_class = bin_drr(row["DRR"])
        c80_class = bin_c80(row["C80"])
        rt60_class = bin_rt60(row["RT60"])

        y = np.array([drr_class, c80_class, rt60_class], dtype=np.int64)

        X.append(x)
        Y.append(y)

    logmels = np.stack(X, axis=0) # shape = (N, 2, F, T)
    labels = np.stack(Y, axis=0)  # shape = (N, 3)

    np.save("BEWO_X_logmel.npy", logmels)
    np.save("BEWO_Y_labels_binned.npy", labels)

    return logmels, labels

def make_pytorch_loader(logmels, labels, batch_size=16, shuffle=True):
    """
    logmels: (N, 2, F, T) - features are already in this shape
    labels:  (N, 3) -> Integer classes
    """
    # Convert directly to Tensor, no need to transpose
    x_tensor = torch.tensor(logmels, dtype=torch.float32)
    # Labels are already properly binned into integers now
    y_tensor = torch.tensor(labels, dtype=torch.long)

    dataset = TensorDataset(x_tensor, y_tensor)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
    )

    return loader

if __name__ == "__main__":
    print("BEWO Data Loader Ready.")
    print("Test physical binning: RT60=1.2s -> Class", bin_rt60(1.2))
