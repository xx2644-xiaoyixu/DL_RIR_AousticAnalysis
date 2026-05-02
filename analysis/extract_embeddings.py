# 用 checkpoint 重新提一遍 embedding（shuffle=False，跟 csv 对齐）
# usage: python analysis/extract_embeddings.py
import os
import sys
import numpy as np
import pandas as pd
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, 'bewo_model'))

from bewo_core import BEWOBackbone
from bewo_data_load import extract_and_save_dataset, load_bewo_arrays, make_pytorch_loader


def main():
    ckpt_path = os.path.join(REPO, 'bewo_outcome', 'bewo_model_best.pth')
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    # checkpoint 是 BEWOWrapper 的 state_dict，里面 backbone 在 conformer.* 这个前缀下
    backbone_sd = {k[len('conformer.'):]: v for k, v in ckpt.items()
                   if k.startswith('conformer.')}

    model = BEWOBackbone(input_freq=128, embed_dim=512)
    model.load_state_dict(backbone_sd, strict=True)
    model.eval()

    label_cols = ['DRR_class', 'C80_class', 'RT60_class', 'ILD_class', 'ITD_class']
    splits = [('train', 'train', 'train'),
              ('validation', 'validation', 'val'),
              ('test', 'test', 'test')]

    for csv_split, audio_split, out_split in splits:
        csv = os.path.join(REPO, 'dataset', 'labels',
                           f'{csv_split}_labels_classification_with_ild_itd.csv')
        feat_dir = os.path.join(REPO, 'bewo_model', 'bewo_features_5lbl')
        x_path, _ = extract_and_save_dataset(csv, audio_split,
                                             os.path.join(REPO, 'dataset', 'audio'),
                                             feat_dir)
        x = np.load(x_path)
        if x.ndim == 4 and x.shape[-1] == 2:
            x = np.transpose(x, (0, 3, 1, 2))

        df = pd.read_csv(csv)
        y = df[label_cols].to_numpy(dtype=np.int64)

        embs = []
        with torch.no_grad():
            for i in range(0, len(x), 32):
                xb = torch.from_numpy(x[i:i + 32].astype(np.float32))
                embs.append(model(xb).numpy())
        emb = np.concatenate(embs)

        np.save(os.path.join(REPO, 'bewo_outcome', f'bewo_embeddings_{out_split}.npy'), emb)
        np.save(os.path.join(REPO, 'bewo_outcome', f'bewo_labels_{out_split}.npy'), y)
        print(f'{out_split}: {emb.shape} saved')


if __name__ == '__main__':
    main()
