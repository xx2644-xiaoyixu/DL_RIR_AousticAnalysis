import os
import json
import torch
import torch.nn as nn
from bl_model_CNN import FrontCNN
from bl_model_core import Conformer
import copy
import numpy as np
import pandas as pd


LABEL_NAMES = ["drr", "c80", "rt60", "ild", "itd"]


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

        self.drr_head = self._make_head(input_dim, hidden_dim, dropout, num_classes)
        self.c80_head = self._make_head(input_dim, hidden_dim, dropout, num_classes)
        self.rt60_head = self._make_head(input_dim, hidden_dim, dropout, num_classes)
        self.ild_head = self._make_head(input_dim, hidden_dim, dropout, num_classes)
        self.itd_head = self._make_head(input_dim, hidden_dim, dropout, num_classes)
        self.heads = nn.ModuleList(
            [
                self.drr_head,
                self.c80_head,
                self.rt60_head,
                self.ild_head,
                self.itd_head,
            ]
        )

    def _make_head(self, input_dim, hidden_dim, dropout, num_classes):
        return nn.Sequential(
            # input shape: (B, D)
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes), 
            # output is (batch, logits), shape is (B, num_classes)    
            # logits are raw class scores before softmax probabilities

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
        ild_logits = self.ild_head(pooled)
        itd_logits = self.itd_head(pooled)

        return drr_logits, c80_logits, rt60_logits, ild_logits, itd_logits

def freeze_conformer_only(model):
    for param in model.conformer.parameters():
        param.requires_grad = False

    for param in model.front_cnn.parameters():
        param.requires_grad = True

    for head in model.heads:
        for param in head.parameters():
            param.requires_grad = True


def _compute_multitask_metrics(logits, y, criterion):
    losses = []
    correct = {}
    loss_values = {}
    total_correct = 0
    batch_size = y.size(0)

    for label_idx, (label_name, label_logits) in enumerate(zip(LABEL_NAMES, logits)):
        label_y = y[:, label_idx]
        label_loss = criterion(label_logits, label_y)
        losses.append(label_loss)
        loss_values[label_name] = label_loss.item()

        # classify
        label_pred = torch.argmax(label_logits, dim=1)
        label_correct = (label_pred == label_y).sum().item()
        correct[label_name] = label_correct
        total_correct += label_correct

    loss = sum(losses)
    return loss, loss_values, correct, total_correct, batch_size


def evaluate_classifier(model, val_loader, criterion, device):
    model.eval()

    total_loss = 0.0
    loss_totals = {label_name: 0.0 for label_name in LABEL_NAMES}

    total_correct = 0
    correct_totals = {label_name: 0 for label_name in LABEL_NAMES}

    total_count = 0
    sample_count = 0

    # don't train validation dataset
    with torch.no_grad():
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device).long()
            batch_size = x.size(0)

            # predicts the logits
            logits = model(x)
            loss, loss_values, correct, batch_correct, _ = _compute_multitask_metrics(
                logits,
                y,
                criterion,
            )

            total_loss += loss.item() * batch_size
            total_correct += batch_correct

            for label_name in LABEL_NAMES:
                loss_totals[label_name] += loss_values[label_name] * batch_size
                correct_totals[label_name] += correct[label_name]

            sample_count += batch_size
            total_count += batch_size * len(LABEL_NAMES)

    metrics = {
        "loss": total_loss / total_count,
        "acc": total_correct / total_count,
    }
    for label_name in LABEL_NAMES:
        metrics[f"{label_name}_loss"] = loss_totals[label_name] / sample_count
        metrics[f"{label_name}_acc"] = correct_totals[label_name] / sample_count

    return metrics

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
        weight_decay=1e-5,
    )

    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_epoch = -1
    best_model_state = None
    best_val_metrics = None

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    for epoch in range(epochs):
        model.train()
        model.conformer.eval()
        
        total_loss = 0.0
        loss_totals = {label_name: 0.0 for label_name in LABEL_NAMES}

        total_correct = 0
        correct_totals = {label_name: 0 for label_name in LABEL_NAMES}

        total_count = 0
        sample_count = 0

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device).long()
            batch_size = x.size(0)

            optimizer.zero_grad()

            logits = model(x)
            loss, loss_values, correct, batch_correct, _ = _compute_multitask_metrics(
                logits,
                y,
                criterion,
            )

            loss.backward()
            optimizer.step()

            total_loss += loss.item() * batch_size
            total_correct += batch_correct

            for label_name in LABEL_NAMES:
                loss_totals[label_name] += loss_values[label_name] * batch_size
                correct_totals[label_name] += correct[label_name]

            sample_count += batch_size
            total_count += batch_size * len(LABEL_NAMES)

        # for all heads together
        train_loss = total_loss / total_count
        train_acc = total_correct / total_count

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"train loss: {train_loss:.2f} | "
            f"train acc: {train_acc:.2f}"
        )
        print(
            f"                 "
            + " | ".join(
                [
                    f"{label_name.upper()} loss/acc: "
                    f"{loss_totals[label_name] / sample_count:.2f}/"
                    f"{correct_totals[label_name] / sample_count:.2f}"
                    for label_name in LABEL_NAMES
                ]
            )
        )

        if val_loader is not None:
            val_metrics = evaluate_classifier(
                model,
                val_loader,
                criterion,
                device,
            )
            val_loss = val_metrics["loss"]
            val_acc = val_metrics["acc"]
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"                 "
                f"val loss: {val_loss:.2f} | "
                f"val acc: {val_acc:.2f}"
            )
            print(
                f"                 "
                + " | ".join(
                    [
                        f"{label_name.upper()} loss/acc: "
                        f"{val_metrics[f'{label_name}_loss']:.2f}/"
                        f"{val_metrics[f'{label_name}_acc']:.2f}"
                        for label_name in LABEL_NAMES
                    ]
                )
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_val_acc = val_acc
                best_epoch = epoch + 1
                best_model_state = copy.deepcopy(model.state_dict())
                best_val_metrics = copy.deepcopy(val_metrics)

                print(
                    f"                 "
                    f"new best model saved at epoch {best_epoch}"
                )

    model.history = history

    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        model.best_epoch = best_epoch
        model.best_val_loss = best_val_loss
        model.best_val_acc = best_val_acc
        model.best_val_metrics = best_val_metrics

        print(
            f"Loaded best model from epoch {best_epoch} | "
            f"best val loss: {best_val_loss:.2f} | "
            f"best val acc: {best_val_acc:.2f}"
        )
        if best_val_metrics is not None:
            print(
                f"                 "
                + " | ".join(
                    [
                        f"{label_name.upper()} val loss/acc: "
                        f"{best_val_metrics[f'{label_name}_loss']:.2f}/"
                        f"{best_val_metrics[f'{label_name}_acc']:.2f}"
                        for label_name in LABEL_NAMES
                    ]
                )
            )

    return model

def test_result(best_baseline_model, test_loader):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    criterion = nn.CrossEntropyLoss()
    
    best_baseline_model.to(device)
    best_baseline_model.eval()
    
    test_metrics = evaluate_classifier(
        best_baseline_model,
        test_loader,
        criterion,
        device,
    )
    
    print(f"Test loss: {test_metrics['loss']:.2f} | Test acc: {test_metrics['acc']:.2f}")
    
    for label_name in LABEL_NAMES:
        print(
            f"{label_name.upper()} test loss/acc: "
            f"{test_metrics[f'{label_name}_loss']:.2f}/"
            f"{test_metrics[f'{label_name}_acc']:.2f}"
        )


def extract_embeddings(model, data_loader, device=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.to(device)
    model.eval()

    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device)

            # 1. CNN frontend
            x = model.front_cnn(x)
            # x shape: (B, T, 128)

            B, T, D = x.shape
            lengths = torch.full(
                (B,),
                T,
                dtype=torch.long,
                device=x.device,
            )

            # 2. Conformer output
            conformer_out, lengths = model.conformer(x, lengths)
            # conformer_out shape: (B, T, 128)

            # 3. Same pooling as classifier forward()
            embedding = conformer_out.mean(dim=1)
            # embedding shape: (B, 128)

            all_embeddings.append(embedding.cpu())
            all_labels.append(y.cpu())

    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()

    return all_embeddings, all_labels




            




