# Part 5 — Analysis

对比 baseline (Conformer, 128-d) 和 bewo (ResNet18-style, 512-d) 两个 encoder 的 embedding，看它们能不能编码 5 个声学参数：DRR、C80、RT60、ILD、ITD。

## 装环境

```
pip install -r requirements.txt
```

## 文件

- `utils.py` — 加载 embedding 和 label 的小函数
- `extract_embeddings.py` — 用 bewo checkpoint 重新提 embedding 的脚本
- `01_decodability.ipynb` — linear probe 看能不能从 embedding 解码出参数（分类 + 回归）
- `02_similarity.ipynb` — embedding cosine similarity 跟参数差的相关性
- `03_visualization.ipynb` — UMAP 可视化
- `figs/` — 跑出来的图

## 跑

三个 notebook 互相独立，按顺序跑或者单跑都行。

```
jupyter notebook
```

打开任意一个 ipynb，菜单 Run > Run All Cells。

## 一些 note

- baseline train embedding 是 shuffle=True 提的，所以行跟 csv 对不上。回归 probe 我用 val 训 test 评，避开了这个问题。
- bewo train 用 checkpoint 重新提了一遍 (`extract_embeddings.py`)，是对齐的。
- ILD/ITD 两个 encoder 都做不好，应该是 mel-spectrogram 把双耳信息丢了。
