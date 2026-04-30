import os
import zipfile
from bewo_model.bewo_data_load import load_dataset, make_pytorch_loader

# 1. Ensure features are extracted
if not os.path.exists("features"):
    print("Extracting features.zip to features/ directory...")
    os.makedirs("features", exist_ok=True)
    with zipfile.ZipFile("features.zip", 'r') as zip_ref:
        zip_ref.extractall("features")
    print("Extraction complete!")

FEATURE_DIR = "./features"
# Handle path if the extracted archive contains another nested features folder
if os.path.exists("./features/features"):
    FEATURE_DIR = "./features/features"
elif os.path.exists("./features/ir_features"):
    FEATURE_DIR = "./features/ir_features"
# Dynamically detect nested folders based on contents
else:
    for item in os.listdir("features"):
        if os.path.isdir(os.path.join("features", item)):
            FEATURE_DIR = os.path.join("features", item)
            break

TRAIN_LABEL_CSV = "./train_labels.csv"

print(f"Using feature directory path: {FEATURE_DIR}")
print("Loading training set (loading only a portion for testing)...")

# 2. Call our custom BEWO data loader
try:
    train_logmels, train_labels = load_dataset(FEATURE_DIR, TRAIN_LABEL_CSV)
    
    if len(train_logmels) == 0:
        print("No data loaded. Please check the feature file path.")
    else:
        train_loader = make_pytorch_loader(train_logmels, train_labels, batch_size=4, shuffle=True)
        
        for x, y in train_loader:
            print("\n✅ Data loader works successfully!")
            print("▶ Input tensor x shape (Binaural BEWO input):", x.shape) 
            print("▶ Label tensor y shape (Discrete classes after physical binning):", y.shape) 
            
            print("\nPrinting true class labels for the first sample in the batch:")
            print(f"  DRR Class: {y[0][0].item()} (0: Far field/Low direct, 1: Transition, 2: Near field/High direct)")
            print(f"  C80 Class: {y[0][1].item()} (0: Poor clarity, 1: Fair, 2: High clarity)")
            print(f"  RT60 Class: {y[0][2].item()} (0: Short reverb, 1: Medium reverb, 2: Long reverb)")
            break
except Exception as e:
    print("An error occurred:", e)
