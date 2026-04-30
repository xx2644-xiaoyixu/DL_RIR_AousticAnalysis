import torch
import torch.nn as nn
import copy
from bewo_model.bewo_core import BEWOBackbone

class BEWOProbeClassifier(nn.Module):
    def __init__(self, bewo_backbone, embed_dim=256, num_classes=3):
        super().__init__()
        
        self.bewo = bewo_backbone
        
        # Linear Probes
        # To ensure "strict" linear probing, the probe heads should be as simple as possible (single Linear layer),
        # measuring whether the Embedding itself disentangles spatial information.
        self.drr_probe = nn.Linear(embed_dim, num_classes)
        self.c80_probe = nn.Linear(embed_dim, num_classes)
        self.rt60_probe = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        """
        x: [B, 2, F, T]
        """
        # BEWO feature extraction
        embedding = self.bewo(x) # [B, embed_dim]
        
        # Classification via probes
        drr_logits = self.drr_probe(embedding)
        c80_logits = self.c80_probe(embedding)
        rt60_logits = self.rt60_probe(embedding)
        
        return drr_logits, c80_logits, rt60_logits

def evaluate_probes(model, val_loader, criterion, device):
    model.eval()
    
    total_loss = 0.0
    drr_correct, c80_correct, rt60_correct = 0, 0, 0
    total_count = 0

    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).long() # y: [B, 3]

            drr_y, c80_y, rt60_y = y[:, 0], y[:, 1], y[:, 2]

            drr_logits, c80_logits, rt60_logits = model(x)

            loss = criterion(drr_logits, drr_y) + \
                   criterion(c80_logits, c80_y) + \
                   criterion(rt60_logits, rt60_y)

            total_loss += loss.item() * x.size(0)

            drr_correct += (torch.argmax(drr_logits, dim=1) == drr_y).sum().item()
            c80_correct += (torch.argmax(c80_logits, dim=1) == c80_y).sum().item()
            rt60_correct += (torch.argmax(rt60_logits, dim=1) == rt60_y).sum().item()
            
            total_count += x.size(0)

    avg_loss = total_loss / total_count
    drr_acc = drr_correct / total_count
    c80_acc = c80_correct / total_count
    rt60_acc = rt60_correct / total_count

    return avg_loss, drr_acc, c80_acc, rt60_acc

def train_bewo_probes(
    model,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.to(device)

    # !!! STRICT LINEAR PROBING: Completely freeze all parameters of the BEWO Backbone !!!
    for param in model.bewo.parameters():
        param.requires_grad = False
    
    # Ensure only the parameters of the three Probe heads are passed to the optimizer
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_model_state = None

    print("Starting Linear Probes training. BEWO Backbone weights are locked.")

    for epoch in range(epochs):
        model.train()
        # Set the frozen part to eval mode (disable Dropout/BatchNorm dynamic updates)
        model.bewo.eval()
        
        total_loss = 0.0
        
        for x, y in train_loader:
            x, y = x.to(device), y.to(device).long()
            drr_y, c80_y, rt60_y = y[:, 0], y[:, 1], y[:, 2]

            optimizer.zero_grad()

            drr_logits, c80_logits, rt60_logits = model(x)

            loss = criterion(drr_logits, drr_y) + \
                   criterion(c80_logits, c80_y) + \
                   criterion(rt60_logits, rt60_y)

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)

        train_loss = total_loss / len(train_loader.dataset)
        
        print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {train_loss:.4f}")

        if val_loader is not None:
            val_loss, drr_acc, c80_acc, rt60_acc = evaluate_probes(model, val_loader, criterion, device)
            print(f"            | Val Loss: {val_loss:.4f} | DRR Acc: {drr_acc:.4f} | C80 Acc: {c80_acc:.4f} | RT60 Acc: {rt60_acc:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = copy.deepcopy(model.state_dict())
                print(f"            | -> New best model saved!")

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print("Training finished. Loaded best model.")

    return model
