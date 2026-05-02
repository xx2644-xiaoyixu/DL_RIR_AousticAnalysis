import os
import sys
# Add parent directory to path so it can find bewo_model when run from inside bewo_outcome
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from bewo_model.bewo_data_load import extract_and_save_dataset, load_bewo_arrays, make_pytorch_loader
from bewo_model.bewo_core import BEWOBackbone

# Import Member 2's probe training and evaluation functions to align with the rest of the team
import sys
sys.path.append('./baseline_model')
from bl_model_classification import train_frontCNN_probes, evaluate_classifier

class BEWOWrapper(nn.Module):
    """
    Wrapper designed to mirror the structure expected by Member 2's `train_frontCNN_probes`.
    This ensures we strictly reuse their metrics and evaluation pipelines.
    """
    def __init__(self, bewo_backbone, embed_dim=512, num_classes=3):
        super().__init__()
        # We assign the BEWO backbone to 'conformer' so Member 2's training loop freezes it properly.
        self.conformer = bewo_backbone
        # 'front_cnn' is expected by their loop to unfreeze. Identity has no params.
        self.front_cnn = nn.Identity()

        # Reusing the exact head designs from Member 2's RIRConformerClassifier
        hidden_dim = 64
        dropout = 0.1
        
        self.drr_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.c80_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.rt60_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.ild_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        self.itd_head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self.heads = nn.ModuleList([
            self.drr_head, 
            self.c80_head, 
            self.rt60_head, 
            self.ild_head, 
            self.itd_head
        ])

    def forward(self, x):
        # bewo_core.py returns the pooled embedding directly
        pooled = self.conformer(x)
        return self.drr_head(pooled), self.c80_head(pooled), self.rt60_head(pooled), self.ild_head(pooled), self.itd_head(pooled)

def main():
    AUDIO_DIR = "./dataset/audio"
    BEWO_FEAT_DIR = "./bewo_model/bewo_features"
    TRAIN_CSV = "./dataset/labels/train_labels_classification_with_ild_itd.csv"
    VAL_CSV = "./dataset/labels/validation_labels_classification_with_ild_itd.csv"
    TEST_CSV = "./dataset/labels/test_labels_classification_with_ild_itd.csv"
    
    print(">>> Step 1: BEWO Preprocessing audio to image (Log-Mel)")
    # Extract features natively (or skip if already extracted to avoid redundancy)
    tr_x_path, tr_y_path = extract_and_save_dataset(TRAIN_CSV, "train", AUDIO_DIR, BEWO_FEAT_DIR)
    va_x_path, va_y_path = extract_and_save_dataset(VAL_CSV, "validation", AUDIO_DIR, BEWO_FEAT_DIR)
    te_x_path, te_y_path = extract_and_save_dataset(TEST_CSV, "test", AUDIO_DIR, BEWO_FEAT_DIR)
    
    print(">>> Step 2: Loading loaders")
    train_x, train_y = load_bewo_arrays(tr_x_path, tr_y_path)
    val_x, val_y = load_bewo_arrays(va_x_path, va_y_path)
    test_x, test_y = load_bewo_arrays(te_x_path, te_y_path)

    train_loader = make_pytorch_loader(train_x, train_y, batch_size=32, shuffle=True)
    val_loader = make_pytorch_loader(val_x, val_y, batch_size=32, shuffle=False)
    test_loader = make_pytorch_loader(test_x, test_y, batch_size=32, shuffle=False)

    print(">>> Step 3: Initialize BEWO Backbone & Reused Probe")
    backbone = BEWOBackbone(input_freq=train_x.shape[2], embed_dim=512)
    probe_model = BEWOWrapper(backbone, embed_dim=512, num_classes=3)

    print(">>> Step 4: Train using Member 2's loop and metrics")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Reusing Member 2's train_frontCNN_probes
    trained_model = train_frontCNN_probes(
        probe_model, train_loader, val_loader, 
        epochs=15, lr=1e-3, device=device
    )
    
    print(">>> Step 5: Test Evaluation & Prediction Output")
    # Reusing Member 2's evaluate_classifier
    criterion = nn.CrossEntropyLoss()
    test_metrics = evaluate_classifier(trained_model, test_loader, criterion, device)
    test_loss = test_metrics["loss"]
    test_acc = test_metrics["acc"]
    
    print(f"=========================================")
    print(f"BEWO Final Test Predictions Results:")
    print(f"Overall Test Loss: {test_loss:.4f}")
    print(f"Overall Test Accuracy: {test_acc*100:.2f}%")
    print(f"=========================================")

    print(">>> Step 5.5: Saving the best model state_dict")
    model_save_path = "bewo_outcome/bewo_model_best.pth"
    torch.save(trained_model.state_dict(), model_save_path)
    print(f"✅ Saved trained model to {model_save_path}")

    print(">>> Step 6: Extracting 512D Embeddings for Member 4 Analysis")
    trained_model.eval()
    
    splits = [
        ("train", train_loader),
        ("val", val_loader),
        ("test", test_loader)
    ]
    
    import numpy as np
    
    for split_name, loader in splits:
        all_embeddings = []
        all_labels = []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                # Pass through the BEWO backbone natively
                embeddings = trained_model.conformer(x)
                all_embeddings.append(embeddings.cpu().numpy())
                all_labels.append(y.numpy())
                
        final_embeddings = np.concatenate(all_embeddings, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        
        emb_path = f"bewo_outcome/bewo_embeddings_{split_name}.npy"
        lbl_path = f"bewo_outcome/bewo_labels_{split_name}.npy"
        np.save(emb_path, final_embeddings)
        np.save(lbl_path, final_labels)
        
        print(f"✅ Successfully extracted and saved {len(final_embeddings)} embeddings for '{split_name}' (Dim: {final_embeddings.shape[1]})")

if __name__ == "__main__":
    main()
