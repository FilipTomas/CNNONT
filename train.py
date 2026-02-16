# %%
import glob
import math
import copy
import numpy as np
import torch
import torch.nn as nn
import model as md
from torch.utils.data import Dataset, DataLoader

print("torch cuda build:", torch.version.cuda)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

# Optional: cap GPU memory usage
if device.type == "cuda":
    torch.cuda.set_per_process_memory_fraction(1/2, device=0)

# ----------------------------
# Config
# ----------------------------
MAX_EPOCHS = 200
PATIENCE = 14          # early stop if val class-1 F1 doesn't improve for N epochs
MIN_DELTA = 1e-4       # require at least this much improvement in F1
BEST_PATH = "data/models_small_batch_sepconv/cnnont_model_best.pth"
LAST_PATH = "data/models_small_batch_sepconv/cnnont_model_last.pth"

# IMPORTANT: class with label 1 is your "class 2"
TARGET_CLASS = 1
IGNORE_INDEX = -100

# Paths to your pre-batched NPZ directories
TRAIN_ROOT = "/mnt/share1_Jabba/ftomas/piles/25x_small_batches/train"
VAL_ROOT   = "/mnt/share1_Jabba/ftomas/piles/25x_small_batches/val"
TEST_ROOT  = "/mnt/share1_Jabba/ftomas/piles/25x_small_batches/test"

# Global features (per-batch here; if you later want per-read g, store it in NPZ)
g = torch.tensor([[50, 25]], dtype=torch.float32, device=device)

# ----------------------------
# Dataset: each item is one saved batch (.npz)
# ----------------------------
class NPZBatchDataset(Dataset):
    def __init__(self, root_dir: str):
        self.files = sorted(glob.glob(f"{root_dir}/bucket_*/batch_*.npz"))
        if not self.files:
            raise RuntimeError(f"No .npz batches found under: {root_dir}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        data = np.load(self.files[idx], allow_pickle=True)
        X = torch.from_numpy(data["X"]).unsqueeze(1)  # [B, 1, L]
        Y = torch.from_numpy(data["Y"]).long()        # [B, L] with IGNORE_INDEX padding
        lengths = torch.from_numpy(data["lengths"]).long()
        return X, Y, lengths


train_loader = DataLoader(
    NPZBatchDataset(TRAIN_ROOT),
    batch_size=1, shuffle=True,
    num_workers=4, pin_memory=True,
)

val_loader = DataLoader(
    NPZBatchDataset(VAL_ROOT),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True,
)

test_loader = DataLoader(
    NPZBatchDataset(TEST_ROOT),
    batch_size=1, shuffle=False,
    num_workers=4, pin_memory=True,
)

# ----------------------------
# Model / optimizer / loss
# ----------------------------
model = md.CNNONT(2, k = 7, film_hidden= 64, hidden= 64, dilation_levels=7, dilation_step=1, n_classes=3).to(device)
print("Model parameters:", sum(p.numel() for p in model.parameters()))
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-5)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",          # because we maximize F1
    factor=0.5,          # halve LR when plateaued
    patience=3,          # epochs with no improvement
    threshold=1e-4,
    min_lr=1e-6,
    verbose=True
)

# IMPORTANT: ignore padding labels (-100)
class_w = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32, device=device)
criterion = md.FocalLoss(alpha=class_w, gamma = 0, ignore_index=IGNORE_INDEX).to(device)

# Optional AMP
use_amp = False #(device.type == "cuda")
scaler = torch.amp.GradScaler("cuda") if use_amp else None


# ----------------------------
# Metrics + confusion matrix
# ----------------------------
def eval_target_metrics(model, loader, g, device, target_class=1, ignore_index=-100):
    """
    Computes precision/recall/F1 for target_class over all non-padding positions.
    """
    model.eval()
    tp = fp = fn = 0

    with torch.no_grad():
        for X, Y, _ in loader:
            X = X.squeeze(0).to(device, non_blocking=True)  # [B,1,L]
            Y = Y.squeeze(0).to(device, non_blocking=True)  # [B,L]

            logits = model(X, g)                              # [B,C,L]
            pred = torch.argmax(logits, dim=1)                # [B,L]

            mask = (Y != ignore_index)
            y = Y[mask]
            p = pred[mask]

            tp += int(((p == target_class) & (y == target_class)).sum().item())
            fp += int(((p == target_class) & (y != target_class)).sum().item())
            fn += int(((p != target_class) & (y == target_class)).sum().item())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def confusion_matrix_stream(model, loader, g, device, num_classes=3, ignore_index=-100):
    cm = torch.zeros(num_classes, num_classes, dtype=torch.long, device=device)
    model.eval()

    with torch.no_grad():
        for X, Y, _ in loader:
            X = X.squeeze(0).to(device, non_blocking=True)  # [B,1,L]
            Y = Y.squeeze(0).to(device, non_blocking=True)  # [B,L]

            logits = model(X, g)                 # [B,C,L]
            pred = logits.argmax(dim=1)          # [B,L]

            mask = (Y != ignore_index)
            y = Y[mask].to(torch.long)           # [N]
            p = pred[mask].to(torch.long)        # [N]

            # flatten pair (y,p) into single index: idx = y*num_classes + p
            idx = y * num_classes + p            # [N]
            cm += torch.bincount(idx, minlength=num_classes*num_classes).view(num_classes, num_classes)

    return cm.cpu()



# %%
# ----------------------------
# Training loop with early stopping on VAL class-1 F1
# ----------------------------
print("Starting training...")

best_f1 = -math.inf
best_state = None
bad_epochs = 0

for epoch in range(MAX_EPOCHS):
    model.train()
    running_loss = 0.0
    steps = 0

    for X, Y, _ in train_loader:
        X = X.squeeze(0).to(device, non_blocking=True)  # [B,1,L]
        Y = Y.squeeze(0).to(device, non_blocking=True)  # [B,L]

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                logits = model(X, g)

            # Safety: catch AMP overflow early
            if not torch.isfinite(logits).all():
                print("Non-finite logits detected (AMP overflow).")
                print("X min/max:", float(X.min()), float(X.max()))
                print("logits min/max:", float(logits.min()), float(logits.max()))
                torch.save({"X": X.detach().cpu(), "Y": Y.detach().cpu()}, "bad_batch.pt")
                raise RuntimeError("Stopping due to non-finite logits")

            # compute loss in fp32 for stability
            loss = criterion(logits.float(), Y)

            if not torch.isfinite(loss):
                print("Non-finite loss detected.")
                torch.save({"X": X.detach().cpu(), "Y": Y.detach().cpu()}, "bad_batch.pt")
                raise RuntimeError("Stopping due to non-finite loss")

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()

        else:
            logits = model(X, g)
            loss = criterion(logits, Y)

            if not torch.isfinite(loss):
                print("Non-finite loss detected (fp32 path).")
                print("X min/max:", float(X.min()), float(X.max()))
                print("logits min/max:", float(logits.min()), float(logits.max()))
                torch.save({"X": X.detach().cpu(), "Y": Y.detach().cpu()}, "bad_batch.pt")
                raise RuntimeError("Stopping due to non-finite loss")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Update loss stats (for BOTH branches)
        running_loss += float(loss.item())
        steps += 1

    train_loss = running_loss / max(1, steps)

    # ---- Validate target metrics on VAL split
    p1, r1, f1 = eval_target_metrics(
        model, val_loader, g, device,
        target_class=TARGET_CLASS,
        ignore_index=IGNORE_INDEX
    )

    print(
        f"Epoch {epoch+1}/{MAX_EPOCHS} | "
        f"train_loss={train_loss:.6f} | "
        f"val(class={TARGET_CLASS}) P={p1:.4f} R={r1:.4f} F1={f1:.4f}"
    )
    scheduler.step(f1)

    # ---- Early stopping on F1
    if f1 > best_f1 + MIN_DELTA:
        best_f1 = f1
        best_state = copy.deepcopy(model.state_dict())
        torch.save(best_state, BEST_PATH)
        bad_epochs = 0
        print(f"  ↳ new best F1={best_f1:.4f} (saved {BEST_PATH})")
    else:
        bad_epochs += 1
        print(f"  ↳ no improvement for {bad_epochs}/{PATIENCE} epochs")
        if bad_epochs >= PATIENCE:
            print("Early stopping triggered.")
            break

# Save last state too
torch.save(model.state_dict(), LAST_PATH)
print(f"Saved last model to {LAST_PATH}")

# Load best model before final testing
if best_state is not None:
    model.load_state_dict(best_state)
    print(f"Loaded best model with val class-{TARGET_CLASS} F1={best_f1:.4f}")

# %%
# ----------------------------
# Final evaluation on TEST split
# ----------------------------
p1, r1, f1 = eval_target_metrics(
    model, test_loader, g, device,
    target_class=TARGET_CLASS,
    ignore_index=IGNORE_INDEX
)
print(f"TEST (class={TARGET_CLASS}) P={p1:.4f} R={r1:.4f} F1={f1:.4f}")

cm = confusion_matrix_stream(model, test_loader, g, device, num_classes=3, ignore_index=IGNORE_INDEX)
print("Confusion matrix (TEST):")
print(cm)
