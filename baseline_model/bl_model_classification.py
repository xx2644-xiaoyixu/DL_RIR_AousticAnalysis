import torch
import torch.nn as nn
from bl_model_CNN import FrontCNN
from bl_model_core import Conformer
import copy


class RIRConformerClassifier(nn.Module):
    def __init__(
        self,
        frontcnn,
        conformer,
        input_dim=128,
        hidden_dim=64,
        dropout=0.1,
        num_classes=3,
    ):
        super().__init__()

        self.front_cnn = frontcnn
        # let model recognize conformer
        self.conformer = conformer

        self.drr_head = nn.Sequential(
            # input shape: (B, D)
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes), 
            # output is (batch, logits), shape is (B, num_classes)    
            # logits = 模型还没转成概率之前的类别分数, softmax(logits) = 概率

        )

        self.c80_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

        self.rt60_head = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x, lengths=None):
        """
        x: (B, 2, F, T), from data_loader
        """

        x = self.front_cnn(x)
        # x: (B, T, 128)

        B, T, D = x.shape

        # if the length of x[2] is all same -> None
        if lengths is None:
            lengths = torch.full((B,), T, dtype=torch.long, device=x.device)

        x, lengths = self.conformer(x, lengths)
        # x: (B, T, 128)

        pooled = x.mean(dim=1)

        drr_logits = self.drr_head(pooled)
        c80_logits = self.c80_head(pooled)
        rt60_logits = self.rt60_head(pooled)

        return drr_logits, c80_logits, rt60_logits

def freeze_conformer_only(model):
    for param in model.conformer.parameters():
        param.requires_grad = False

    for param in model.front_cnn.parameters():
        param.requires_grad = True

    for head in [model.drr_head, model.c80_head, model.rt60_head]:
        for param in head.parameters():
            param.requires_grad = True

def evaluate_classifier(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_count = 0

    # don't train validation dataset
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).long()

            drr_y = y[:, 0]
            c80_y = y[:, 1]
            rt60_y = y[:, 2]

            drr_logits, c80_logits, rt60_logits = model(x)

            loss_drr = criterion(drr_logits, drr_y)
            loss_c80 = criterion(c80_logits, c80_y)
            loss_rt60 = criterion(rt60_logits, rt60_y)

            loss = loss_drr + loss_c80 + loss_rt60

            total_loss += loss.item() * x.size(0)

            drr_pred = torch.argmax(drr_logits, dim=1)
            c80_pred = torch.argmax(c80_logits, dim=1)
            rt60_pred = torch.argmax(rt60_logits, dim=1)

            correct = (
                (drr_pred == drr_y).sum()
                + (c80_pred == c80_y).sum()
                + (rt60_pred == rt60_y).sum()
            )

            total_correct += correct.item()
            total_count += x.size(0) * 3

    val_loss = total_loss / len(val_loader.dataset)
    val_acc = total_correct / total_count

    return val_loss, val_acc


def train_frontCNN_probes(
    model,
    train_loader,
    val_loader=None,
    epochs=20,
    lr=1e-3,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    model.to(device)

    # to make sure freeze conformer
    freeze_conformer_only(model)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        model.conformer.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()

            drr_y = y[:, 0]
            c80_y = y[:, 1]
            rt60_y = y[:, 2]

            # 把上一轮 batch 留下来的梯度清零
            optimizer.zero_grad()

            drr_logits, c80_logits, rt60_logits = model(x)

            loss_drr = criterion(drr_logits, drr_y)
            loss_c80 = criterion(c80_logits, c80_y)
            loss_rt60 = criterion(rt60_logits, rt60_y)

            loss = loss_drr + loss_c80 + loss_rt60

            loss.backward()
            optimizer.step()

            # 根据每个batch的样本数量加权计算平均loss而不是简单平均
            total_loss += loss.item() * x.size(0)

            drr_pred = torch.argmax(drr_logits, dim=1)
            c80_pred = torch.argmax(c80_logits, dim=1)
            rt60_pred = torch.argmax(rt60_logits, dim=1)

            correct = (
                (drr_pred == drr_y).sum()
                + (c80_pred == c80_y).sum()
                + (rt60_pred == rt60_y).sum()
            )

            total_correct += correct.item()
            total_count += x.size(0) * 3

        # for all heads together
        train_loss = total_loss / len(train_loader.dataset)
        train_acc = total_correct / total_count

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {train_loss:.4f} | "
            f"train acc: {train_acc:.4f}"
        )

        if val_loader is not None:
            val_loss, val_acc = evaluate_classifier(
                model,
                val_loader,
                criterion,
                device,
            )

            print(
                f"                 "
                f"val loss: {val_loss:.4f} | "
                f"val acc: {val_acc:.4f}"
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())

                print(
                    f"                 "
                    f"new best model saved at epoch {best_epoch}"
                )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

        print(
            f"Loaded best model from epoch {best_epoch} | "
            f"best val loss: {best_val_loss:.4f} | "
            f"best val acc: {best_val_acc:.4f}"
        )

    return model





            

