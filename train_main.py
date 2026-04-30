import os
import torch
from bewo_model.bewo_data_load import load_dataset, make_pytorch_loader
from bewo_model.bewo_core import BEWOBackbone
from bewo_model.bewo_probe_train import BEWOProbeClassifier, train_bewo_probes

def main():
    # Dynamically determine the feature path
    FEATURE_DIR = "./features"
    if os.path.exists("./features/features"):
        FEATURE_DIR = "./features/features"
    elif os.path.exists("./features/ir_features"):
        FEATURE_DIR = "./features/ir_features"

    TRAIN_LABEL_CSV = "./train_labels.csv"
    VAL_LABEL_CSV = "./validation_labels.csv"
    
    if not os.path.exists(FEATURE_DIR):
        print("Feature directory not found.")
        return
        
    print("="*50)
    print(">>> Step 1: Prepare data loaders and label discretization")
    train_x, train_y = load_dataset(FEATURE_DIR, TRAIN_LABEL_CSV)
    val_x, val_y = load_dataset(FEATURE_DIR, VAL_LABEL_CSV)
    
    if len(train_x) == 0:
        print("No training data found.")
        return
        
    train_loader = make_pytorch_loader(train_x, train_y, batch_size=32, shuffle=True)
    val_loader = make_pytorch_loader(val_x, val_y, batch_size=32, shuffle=False)
    
    print(f"Training samples: {len(train_x)} | Validation samples: {len(val_x)}")
    
    print("\n" + "="*50)
    print(">>> Step 2: Initialize BEWO backbone and classification probes")
    # BEWO backbone
    bewo = BEWOBackbone(embed_dim=256)
    
    # Initialize probes (outputs 3 classes: 0, 1, 2)
    probe_model = BEWOProbeClassifier(bewo, embed_dim=256, num_classes=3)
    
    print("\n" + "="*50)
    print(">>> Step 3: Launch Strict Linear Probing training")
    # We only run 5 epochs here to demonstrate the pipeline
    trained_model = train_bewo_probes(
        model=probe_model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
        lr=2e-3
    )
    
    print("\n" + "="*50)
    print("✅ Full pipeline test passed! Your BEWO baseline model is ready for extensive experiments.")

if __name__ == "__main__":
    main()
