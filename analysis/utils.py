# 加载 embedding 和 label 用的小工具
import os
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABEL_DIR = os.path.join(REPO, 'dataset', 'labels')

TARGET_NAMES = ['DRR', 'C80', 'RT60', 'ILD', 'ITD']


def _emb_path(encoder, split):
    if encoder == 'baseline':
        return os.path.join(REPO, 'baseline_outcome', f'{split}_embeddings.npy')
    if encoder == 'bewo':
        return os.path.join(REPO, 'bewo_outcome', f'bewo_embeddings_{split}.npy')
    raise ValueError(encoder)


def _lbl_path(encoder, split):
    if encoder == 'baseline':
        return os.path.join(REPO, 'baseline_outcome', f'{split}_labels.npy')
    if encoder == 'bewo':
        return os.path.join(REPO, 'bewo_outcome', f'bewo_labels_{split}.npy')
    raise ValueError(encoder)


def load_embeddings(encoder, split):
    emb = np.load(_emb_path(encoder, split)).astype(np.float32)
    lbl = np.load(_lbl_path(encoder, split)).astype(np.int64)
    # 兼容只有 3 列的旧 label 文件（DRR/C80/RT60）
    if lbl.shape[1] == 3:
        pad = np.full((lbl.shape[0], 2), -1, dtype=np.int64)
        lbl = np.concatenate([lbl, pad], axis=1)
    return emb, lbl


def load_continuous_labels(split):
    csv_map = {
        'train': 'train_labels_classification_with_ild_itd.csv',
        'val': 'validation_labels_classification_with_ild_itd.csv',
        'test': 'test_labels_classification_with_ild_itd.csv',
    }
    df = pd.read_csv(os.path.join(LABEL_DIR, csv_map[split]))
    cont = ['DRR', 'C80', 'RT60', 'ILD', 'ITD_ms']
    cls = ['DRR_class', 'C80_class', 'RT60_class', 'ILD_class', 'ITD_class']
    return df[['id'] + cont + cls].reset_index(drop=True)


def is_csv_aligned(class_labels, df):
    # 检查 .npy label 跟 csv 行是否对得上
    cls_cols = ['DRR_class', 'C80_class', 'RT60_class', 'ILD_class', 'ITD_class']
    expected = df[cls_cols].to_numpy(dtype=np.int64)
    if class_labels.shape != expected.shape:
        return False
    return bool(np.array_equal(class_labels, expected))
