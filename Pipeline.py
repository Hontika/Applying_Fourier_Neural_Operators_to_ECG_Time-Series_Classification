#!/usr/bin/env python
# coding: utf-8

# # U-net

# ## Imports

# In[1]:


# In[4]:

import os, json, time, hashlib, shutil
from pathlib import Path
from itertools import product
import random  # NEW

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from sklearn.metrics import confusion_matrix, classification_report

# --------------------------
# Reproducible seeding
# --------------------------
SEED = 42  # or any integer you like

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    # Make cuDNN deterministic (slightly slower but reproducible)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)
print(f"Global seed set to: {SEED}")

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)
if device == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ## Basic Data Loading

# In[2]:


from pathlib import Path
import numpy as np

# DATASET_NAME = "ECG5000"
DATASET_NAME = "NonInvasiveFetalECGThorax1"
# DATASET_NAME = "NonInvasiveFetalECGThorax2"

# Robust path resolution (handles running notebook from different cwd)
HERE = Path().resolve()
DATA_ROOT = HERE / "Data"
if not DATA_ROOT.exists():
    # fallback: search one level up (common if notebook is in subfolder)
    if (HERE.parent / "Data").exists():
        DATA_ROOT = HERE.parent / "Data"

DATA_DIR = DATA_ROOT / DATASET_NAME
assert DATA_DIR.exists(), f"Dataset folder not found: {DATA_DIR}"

train_path = DATA_DIR / f"{DATASET_NAME}_TRAIN.ts"
test_path  = DATA_DIR / f"{DATASET_NAME}_TEST.ts"
assert train_path.exists(), f"Missing: {train_path}"
assert test_path.exists(), f"Missing: {test_path}"

print("Using dataset:", DATASET_NAME)
print("Train file:", train_path)
print("Test file :", test_path)

def load_ucr_ts(ts_path: Path):
    """
    Minimal UCR/UEA .ts loader for univariate classification datasets.
    Returns: X (N, L) float32, y_raw (N,) as original labels (strings)
    """
    X, y = [], []
    in_data = False

    with ts_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            low = line.lower()

            # header control
            if low.startswith("@data"):
                in_data = True
                continue
            if not in_data:
                continue

            # data lines:
            # format typically: series_values:label
            # where series_values are comma-separated floats
            if ":" not in line:
                continue

            series_part, label_part = line.rsplit(":", 1)
            label = label_part.strip()
            series_str = series_part.strip()

            # Handle optional parentheses/whitespace
            # Values may be like "0.1,0.2,..." (univariate)
            vals = [v for v in series_str.split(",") if v != ""]
            vals = np.array(vals, dtype=np.float32)

            X.append(vals)
            y.append(label)

    X = np.stack(X, axis=0).astype(np.float32)
    y = np.array(y)

    return X, y

# Load raw labels
X_train_full, y_train_raw = load_ucr_ts(train_path)
X_test, y_test_raw        = load_ucr_ts(test_path)

# Map labels -> 0..K-1 consistently using TRAIN classes
unique_labels = sorted(np.unique(y_train_raw).tolist())
label_to_idx = {lab: i for i, lab in enumerate(unique_labels)}

y_train_full = np.array([label_to_idx[lab] for lab in y_train_raw], dtype=np.int64)
y_test       = np.array([label_to_idx.get(lab, -1) for lab in y_test_raw], dtype=np.int64)

# Sanity check: test labels must all exist in training label set
assert (y_test >= 0).all(), "Found test labels not present in training labels."

classes = unique_labels  # for plotting
num_classes = len(classes)
seq_len = X_train_full.shape[1]

print("Train shape:", X_train_full.shape, "Test shape:", X_test.shape)
print("Classes:", classes, "| num_classes:", num_classes, "| seq_len:", seq_len)


# X_train_full, y_train_full, X_test, y_test already loaded above

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.2,
    random_state=42,
    stratify=y_train_full
)


# ## Normalize per sample

# In[3]:


def zscore_per_sample(X, eps=1e-8):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + eps)

X_train = zscore_per_sample(X_train)
X_val   = zscore_per_sample(X_val)
X_test  = zscore_per_sample(X_test)


# ## Pytorch Datalset/Dataloader

# In[4]:


class ECGDataset(Dataset):
    def __init__(self, X, y, augment=False, max_shift=0):
        """
        X: np.array, shape (N, L)
        y: np.array, shape (N,)
        augment: if True, apply random time-shift
        max_shift: maximum absolute shift (in samples). 0 disables shifting.
        """
        self.X = torch.from_numpy(X).unsqueeze(1)  # (N, 1, L)
        self.y = torch.from_numpy(y).long()
        self.augment = augment
        self.max_shift = int(max_shift)

    def __len__(self):
        return len(self.X)

    def _random_shift(self, x):
        """
        x: tensor of shape (C, L)
        Returns: shifted tensor with zero-padding (no wrap-around).
        """
        if self.max_shift <= 0:
            return x

        # Sample shift uniformly in [-max_shift, max_shift]
        shift = torch.randint(-self.max_shift, self.max_shift + 1, (1,)).item()
        if shift == 0:
            return x

        L = x.shape[-1]

        if shift > 0:
            # Shift to the right: new values at the left are zero
            pad = torch.zeros_like(x[..., :shift])
            x_shifted = torch.cat([pad, x[..., :-shift]], dim=-1)
        else:
            # Shift to the left: new values at the right are zero
            s = -shift
            pad = torch.zeros_like(x[..., :s])
            x_shifted = torch.cat([x[..., s:], pad], dim=-1)

        return x_shifted

    def __getitem__(self, i):
        x = self.X[i]
        y = self.y[i]

        if self.augment and self.max_shift > 0:
            x = self._random_shift(x)

        return x, y


# ---- Augmentation settings ----
AUG_TIME_SHIFT = False   # set to True for the augmented experiment
MAX_SHIFT = 5            # in samples, e.g. ±5; tune as desired

train_ds = ECGDataset(
    X_train, y_train,
    augment=AUG_TIME_SHIFT,
    max_shift=MAX_SHIFT,
)
val_ds   = ECGDataset(
    X_val,   y_val,
    augment=False,
    max_shift=0,
)
test_ds  = ECGDataset(
    X_test,  y_test,
    augment=False,
    max_shift=0,
)

# Optional: deterministic shuffling if you have SEED defined
# g = torch.Generator()
# g.manual_seed(SEED)

train_loader = DataLoader(
    train_ds,
    batch_size=128,
    shuffle=True,
    drop_last=True,
    # generator=g,
)

val_loader = DataLoader(
    val_ds,
    batch_size=256,
    shuffle=False,
)

test_loader = DataLoader(
    test_ds,
    batch_size=256,
    shuffle=False,
)


# ## Train/Eval utils

# In[5]:


def train_one_epoch(model, loader, opt, criterion, epoch=0):
    model.train()
    total_loss, correct, n = 0.0, 0, 0

    pbar = tqdm(loader, desc=f"Train {epoch:02d}", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        opt.step()

        bs = xb.size(0)
        total_loss += loss.item() * bs
        correct += (logits.argmax(1) == yb).sum().item()
        n += bs

        pbar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.3f}")

    return total_loss/n, correct/n


@torch.no_grad()
def eval_model(model, loader, criterion, epoch=0):
    model.eval()
    total_loss, correct, n = 0.0, 0, 0
    all_preds, all_y = [], []

    pbar = tqdm(loader, desc=f"Eval  {epoch:02d}", leave=False)
    for xb, yb in pbar:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)

        bs = xb.size(0)
        total_loss += loss.item() * bs
        preds = logits.argmax(1)

        correct += (preds == yb).sum().item()
        n += bs

        all_preds.append(preds.cpu().numpy())
        all_y.append(yb.cpu().numpy())

        pbar.set_postfix(loss=f"{total_loss/n:.4f}", acc=f"{correct/n:.3f}")

    return total_loss/n, correct/n, np.concatenate(all_y), np.concatenate(all_preds)


def fit(model, train_loader, val_loader, epochs=40, lr=1e-3, weight_decay=1e-4):
    """
    Train with validation:
    - train_loader: training data
    - val_loader:   validation data (used for model selection)
    """
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc, best_state = 0.0, None
    history = []

    for ep in range(epochs):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, opt, criterion, ep)
        val_loss, val_acc, _, _ = eval_model(model, val_loader, criterion, ep)
        history.append((tr_loss, tr_acc, val_loss, val_acc))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        print(
            f"Epoch {ep:02d}/{epochs-1} | "
            f"train acc {tr_acc:.3f} loss {tr_loss:.4f} | "
            f"val acc {val_acc:.3f} loss {val_loss:.4f}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history




# ## Grad-CAM utils

# In[6]:


import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

class GradCAM1D:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations = out.detach()

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        """
        x: (1, 1, L)
        returns cam: (L,)
        """
        self.model.zero_grad()
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1).item()

        score = logits[:, class_idx]
        score.backward()

        # activations: (1, C, L') gradients: (1, C, L')
        A = self.activations
        dA = self.gradients

        # global-average pool gradients over time to get weights per channel
        weights = dA.mean(dim=-1, keepdim=True)  # (1, C, 1)

        cam = (weights * A).sum(dim=1)  # (1, L')
        cam = F.relu(cam)

        # upsample cam to input length
        cam = F.interpolate(cam.unsqueeze(1), size=x.shape[-1], mode="linear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, class_idx, logits.softmax(1).detach().cpu().numpy()

def show_gradcam_1d(model, dataset, idx, class_names=None, target_layer=None):
    model.eval()
    x, y = dataset[idx]
    x_in = x.unsqueeze(0).to(device)

    # pick default layer if not provided
    if target_layer is None:
        # try to use last conv in decoder (dec1) if exists
        target_layer = getattr(model, "dec1", None)
        if target_layer is not None and hasattr(target_layer, "net"):
            # last Conv1d in that block
            target_layer = target_layer.net[3]
        else:
            # fallback: last Conv1d we can find
            convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv1d)]
            target_layer = convs[-1]

    cam_engine = GradCAM1D(model, target_layer)
    cam, pred_idx, probs = cam_engine(x_in)

    sig = x.squeeze().numpy()
    L = sig.shape[0]

    plt.figure(figsize=(10,3))
    plt.plot(sig, label="ECG")
    plt.fill_between(np.arange(L), sig.min(), sig.max(), where=cam>0,
                     alpha=0.3, step="mid")
    plt.title(
        f"Grad-CAM 1D | true={y.item()} pred={pred_idx} "
        + (f"({class_names[pred_idx]})" if class_names else "")
    )
    plt.legend()
    plt.show()

    print("Class probs:", probs[0])


import torch
import torch.nn.functional as F

def match_length(a, b):
    """
    Make tensors a and b have the same last-dimension length by center-cropping the longer one.
    a, b: (N, C, L)
    Returns: (a_matched, b_matched)
    """
    La = a.shape[-1]
    Lb = b.shape[-1]
    if La == Lb:
        return a, b

    if La > Lb:
        diff = La - Lb
        left = diff // 2
        right = diff - left
        a = a[..., left:La-right]
    else:
        diff = Lb - La
        left = diff // 2
        right = diff - left
        b = b[..., left:Lb-right]

    return a, b



# ## Model 1: Compact 1D U-net

# In[7]:


class ConvBlock1D(nn.Module):
    def __init__(self, cin, cout, k=3, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(cin, cout, k, padding=p),
            nn.BatchNorm1d(cout),
            nn.ReLU(),
            nn.Conv1d(cout, cout, k, padding=p),
            nn.BatchNorm1d(cout),
            nn.ReLU(),
        )
    def forward(self, x): return self.net(x)

class UNet1DClassifier(nn.Module):
    def __init__(self, in_ch=1, base=32, num_classes=5, dropout=0.0):
        super().__init__()
        self.enc1 = ConvBlock1D(in_ch, base)
        self.down1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1D(base, base*2)
        self.down2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1D(base*2, base*4)

        self.up2 = nn.ConvTranspose1d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock1D(base*4, base*2)
        self.up1 = nn.ConvTranspose1d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock1D(base*2, base)

        # classifier head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base, num_classes)
        )

    def forward(self, x):
        e1 = self.enc1(x)      # (B, base, L)
        e2 = self.enc2(self.down1(e1))  # (B, 2base, L/2)
        e3 = self.enc3(self.down2(e2))  # (B, 4base, L/4)

        d2 = self.up2(e3)
        d2, e2 = match_length(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1, e1 = match_length(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.dec1(d1)

        return self.head(d1)


# ## Model 2: 1D U-net + FNO bottleneck

# In[ ]:


class SpectralConv1d(nn.Module):
    """
    1D spectral conv: FFT -> keep modes -> complex linear map -> iFFT
    """
    def __init__(self, in_channels, out_channels, modes):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes, dtype=torch.cfloat))

    def forward(self, x):
        # x: (B, C, L)
        B, C, L = x.shape
        x_ft = torch.fft.rfft(x, dim=-1)  # (B, C, L//2+1)
        out_ft = torch.zeros(B, self.out_channels, x_ft.size(-1), dtype=torch.cfloat, device=x.device)

        m = min(self.modes, x_ft.size(-1))
        out_ft[:, :, :m] = torch.einsum("bci,coi->boi", x_ft[:, :, :m], self.weights[:, :, :m])
        x = torch.fft.irfft(out_ft, n=L, dim=-1)
        return x

class FNOBlock1d(nn.Module):
    def __init__(self, width, modes):
        super().__init__()
        self.spec = SpectralConv1d(width, width, modes)
        self.pw = nn.Conv1d(width, width, 1)
        self.bn = nn.BatchNorm1d(width)
    def forward(self, x):
        x = self.spec(x) + self.pw(x)
        return F.relu(self.bn(x))

class UNetFNO1DClassifier(nn.Module):
    def __init__(self, in_ch=1, base=32, num_classes=5, modes=16, fno_layers=2, dropout=0.0):
        super().__init__()
        self.enc1 = ConvBlock1D(in_ch, base)
        self.down1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1D(base, base*2)
        self.down2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1D(base*2, base*4)

        self.fno = nn.Sequential(*[FNOBlock1d(base*4, modes) for _ in range(fno_layers)])

        self.up2 = nn.ConvTranspose1d(base*4, base*2, 2, stride=2)
        self.dec2 = ConvBlock1D(base*4, base*2)
        self.up1 = nn.ConvTranspose1d(base*2, base, 2, stride=2)
        self.dec1 = ConvBlock1D(base*2, base)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base, num_classes)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.fno(e3)

        d2 = self.up2(b)
        d2, e2 = match_length(d2, e2)
        d2 = torch.cat([d2, e2], dim=1)

        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1, e1 = match_length(d1, e1)
        d1 = torch.cat([d1, e1], dim=1)

        d1 = self.dec1(d1)

        return self.head(d1)


# # Attention U-net

# ## Model 1: Attention U-net (Attention gate + Attention up block)

# In[9]:


import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate1D(nn.Module):
    """
    Attention gate for 1D skip connections.
    g: gating signal from decoder (coarser)
    x: skip features from encoder (finer)
    returns: attended skip features
    """
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv1d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv1d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv1d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm1d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # g and x should have same temporal length here
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # (B,1,L)
        return x * psi       # broadcast channel-wise


class UpBlockAtt1D(nn.Module):
    """
    Upsample, apply attention gate on skip, concat, then ConvBlock1D.
    """
    def __init__(self, F_g, F_l, F_out):
        super().__init__()
        self.up = nn.ConvTranspose1d(F_g, F_out, kernel_size=2, stride=2)
        self.att = AttentionGate1D(F_g=F_out, F_l=F_l, F_int=F_out//2)
        self.conv = ConvBlock1D(F_out + F_l, F_out)

    def forward(self, g, x_skip):
        g_up = self.up(g)
        # in case of odd lengths, align by trimming/padding
        if g_up.size(-1) != x_skip.size(-1):
            minL = min(g_up.size(-1), x_skip.size(-1))
            g_up = g_up[..., :minL]
            x_skip = x_skip[..., :minL]

        x_att = self.att(g_up, x_skip)
        x = torch.cat([g_up, x_att], dim=1)
        return self.conv(x)


# In[10]:


class AttUNet1DClassifier(nn.Module):
    """
    1D Attention U-Net used as feature extractor for classification.
    Attention gates are applied on skip connections.
    """
    def __init__(self, in_ch=1, base=32, num_classes=5, dropout=0.0):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock1D(in_ch, base)
        self.down1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1D(base, base*2)
        self.down2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1D(base*2, base*4)

        # Decoder with attention on skips
        self.up2 = UpBlockAtt1D(F_g=base*4, F_l=base*2, F_out=base*2)
        self.up1 = UpBlockAtt1D(F_g=base*2, F_l=base,   F_out=base)

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base, num_classes)
        )

    def forward(self, x):
        e1 = self.enc1(x)              # (B, base, L)
        e2 = self.enc2(self.down1(e1)) # (B, 2base, L/2)
        e3 = self.enc3(self.down2(e2)) # (B, 4base, L/4)

        d2 = self.up2(e3, e2)          # (B, 2base, L/2)
        d1 = self.up1(d2, e1)          # (B, base, L)

        return self.head(d1)


# ## Model 2: Attention U-net + FNO

# In[11]:


class AttUNetFNO1DClassifier(nn.Module):
    """
    1D Attention U-Net with FNO bottleneck for classification.
    """
    def __init__(self, in_ch=1, base=32, num_classes=5,
                 modes=16, fno_layers=2, dropout=0.0):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock1D(in_ch, base)
        self.down1 = nn.MaxPool1d(2)
        self.enc2 = ConvBlock1D(base, base*2)
        self.down2 = nn.MaxPool1d(2)
        self.enc3 = ConvBlock1D(base*2, base*4)

        # FNO bottleneck (reuse your FNOBlock1d)
        self.fno = nn.Sequential(*[
            FNOBlock1d(base*4, modes) for _ in range(fno_layers)
        ])

        # Decoder with attention skips
        self.up2 = UpBlockAtt1D(F_g=base*4, F_l=base*2, F_out=base*2)
        self.up1 = UpBlockAtt1D(F_g=base*2, F_l=base,   F_out=base)

        # Classification head
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base, num_classes)
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))

        b = self.fno(e3)

        d2 = self.up2(b, e2)
        d1 = self.up1(d2, e1)

        return self.head(d1)


# # ResNet1D for baseline

# In[12]:


class BasicBlock1D(nn.Module):
    """
    Basic ResNet block for 1D signals:
    Conv1d -> BN -> ReLU -> Conv1d -> BN + skip
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv1d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet1DClassifier(nn.Module):
    """
    1D ResNet baseline for ECG classification.
    Roughly analogous to ResNet-18 in 1D.

    base: controls the width of the network (like your U-Net base channels).
    """
    def __init__(self, in_ch=1, base=32, num_classes=5, dropout=0.0):
        super().__init__()

        self.in_planes = base

        # Initial conv: keep it simple and similar scale to U-Net
        self.conv1 = nn.Conv1d(in_ch, base, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(base)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        # ResNet-18 style layer sizes: [base, 2*base, 4*base, 8*base]
        self.layer1 = self._make_layer(BasicBlock1D,  base,      blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock1D,  base * 2,  blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock1D,  base * 4,  blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock1D,  base * 8,  blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base * 8, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # x: (B, 1, L)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)      # (B, C, 1)
        x = torch.flatten(x, 1)  # (B, C)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# # Hyperparameter tuning

# Shared:
# 
# - base (channels): 16, 32, 48, 64
# - lr: 1e-3, 5e-4, 3e-4
# - weight_decay: 0, 1e-5, 1e-4, 5e-4
# - epochs: fixed (30–60); not a search target unless you want early stopping
# 
# Base U-Net-specific:
# 
# - dropout: 0.0, 0.1, 0.2, 0.3 (we’ll add it optionally)
# - kernel_size: 3 or 5 (optional; more invasive)
# 
# FNO-specific:
# 
# - modes: 8, 12, 16, 24, 32
# - fno_layers: 1, 2, 3

# ## Evaluation prep

# In[13]:


def plot_confusion(cm, class_names, normalize=False, title="Confusion"):
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-12)

    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted",
        ylabel="True",
        title=title
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() * 0.6
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i,j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i,j] > thresh else "black")
    plt.show()


def eval_and_plot(model, loader, name="Model"):
    crit = nn.CrossEntropyLoss()
    _, acc, y_true, y_pred = eval_model(model, loader, crit)
    print(f"{name} acc: {acc:.4f}\n")
    print(classification_report(y_true, y_pred, digits=4))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion(cm, [str(c) for c in classes], False, f"{name} Confusion (counts)")
    plot_confusion(cm, [str(c) for c in classes], True,  f"{name} Confusion (norm)")

    return acc


# ## Grid Search

# In[14]:


# Resumable Gridsearch (backend)


import hashlib, json, time
from pathlib import Path
import pandas as pd
import torch
import torch.nn as nn


# ---- Grid definitions ----
grid_shared = {
    "base": [16, 32, 48],
    "lr": [1e-3, 5e-4],
    "weight_decay": [0.0, 1e-4],
    "dropout": [0.0, 0.1, 0.2],
}
grid_fno = {
    "modes": [8, 16, 24],
    "fno_layers": [1, 2],
}


# ---- Model directories ----
MODELS_DIR = Path("models") / DATASET_NAME
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIRS = {
    "base_unet":    MODELS_DIR / "base_unet",
    "unet_fno":     MODELS_DIR / "unet_fno",
    "att_unet":     MODELS_DIR / "att_unet",
    "att_unet_fno": MODELS_DIR / "att_unet_fno",
    "resnet1d":     MODELS_DIR / "resnet1d",   # NEW
}

for d in MODEL_DIRS.values():
    d.mkdir(parents=True, exist_ok=True)

LEADERBOARD_PATH = MODELS_DIR / "leaderboard.csv"


# ---- Helpers ----
def expand_grid(g):
    keys, vals = list(g.keys()), list(g.values())
    return [dict(zip(keys, v)) for v in product(*vals)]


def make_config_key(model_type, cfg):
    items = sorted(cfg.items())
    s = model_type + "|" + "|".join([f"{k}={v}" for k, v in items])
    return hashlib.md5(s.encode("utf-8")).hexdigest()


def load_completed_keys():
    if not LEADERBOARD_PATH.exists():
        return set()
    df = pd.read_csv(LEADERBOARD_PATH, engine="python", on_bad_lines="skip")
    if "config_key" in df.columns:
        return set(df["config_key"].astype(str))
    completed = set()
    # fallback for earlier CSVs without config_key
    use = ["base","lr","weight_decay","dropout","modes","fno_layers"]
    for _, row in df.iterrows():
        cfg = {k: row[k] for k in use if k in row and pd.notna(row[k])}
        completed.add(make_config_key(row["model_type"], cfg))
    return completed


def next_run_dir(model_type):
    parent = MODEL_DIRS[model_type]
    runs = sorted([p for p in parent.glob("run_*") if p.is_dir()])
    rid = len(runs) + 1
    rd = parent / f"run_{rid:03d}"
    rd.mkdir(exist_ok=False)
    return rd, rid


def save_run(model, model_type, cfg, metrics):
    rd, rid = next_run_dir(model_type)

    torch.save(model.state_dict(), rd/"model.pt")
    (rd/"config.json").write_text(json.dumps(cfg, indent=2))
    (rd/"metrics.json").write_text(json.dumps(metrics, indent=2))

    row = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_type": model_type,
        "run_id": rid,
        "config_key": make_config_key(model_type, cfg),
        **cfg,
        **metrics,
    }

    df = pd.DataFrame([row])
    if LEADERBOARD_PATH.exists():
        df.to_csv(LEADERBOARD_PATH, mode="a", header=False, index=False)
    else:
        df.to_csv(LEADERBOARD_PATH, index=False)



# ---- Build model from config ----
def build_model(model_type, cfg):
    if model_type == "base_unet":
        return UNet1DClassifier(1, cfg["base"], num_classes, cfg["dropout"])
    if model_type == "unet_fno":
        return UNetFNO1DClassifier(
            1, cfg["base"], num_classes,
            cfg["modes"], cfg["fno_layers"], cfg["dropout"]
        )
    if model_type == "att_unet":
        return AttUNet1DClassifier(1, cfg["base"], num_classes, cfg["dropout"])
    if model_type == "att_unet_fno":
        return AttUNetFNO1DClassifier(
            1, cfg["base"], num_classes,
            cfg["modes"], cfg["fno_layers"], cfg["dropout"]
        )
    if model_type == "resnet1d":  # NEW
        return ResNet1DClassifier(
            in_ch=1,
            base=cfg["base"],
            num_classes=num_classes,
            dropout=cfg["dropout"],
        )
    raise ValueError(model_type)




# ---- Run one config ----
def run_one(model_type, cfg, epochs=40):
    model = build_model(model_type, cfg)

    # Train with validation
    model, _ = fit(
        model,
        train_loader,
        val_loader,  # IMPORTANT: use validation loader here
        epochs=epochs,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    crit = nn.CrossEntropyLoss()

    # Validation metrics (for logging / model selection insight)
    val_loss, val_acc, _, _ = eval_model(model, val_loader, crit)

    # Test metrics (for final performance)
    test_loss, test_acc, _, _ = eval_model(model, test_loader, crit)

    metrics = {
        "val_acc": float(val_acc),
        "test_acc": float(test_acc),
        "epochs": epochs,
        "params_million": sum(p.numel() for p in model.parameters()) / 1e6,
    }

    save_run(model, model_type, cfg, metrics)
    return metrics





# ---- Configs per model type ----
def get_cfgs_for_type(model_type):
    shared = expand_grid(grid_shared)
    if model_type in ["base_unet", "att_unet", "resnet1d"]:
        # models without FNO-specific params
        return shared
    else:  # FNO models
        fno = expand_grid(grid_fno)
        cfgs = []
        for s in shared:
            for f in fno:
                c = dict(s); c.update(f); cfgs.append(c)
        return cfgs



def count_remaining_configs(model_type):
    done = load_completed_keys()
    cfgs = get_cfgs_for_type(model_type)
    return sum(1 for cfg in cfgs if make_config_key(model_type, cfg) not in done)



# ---- Resumable Gridsearch (widget compatible) ----
def grid_search_resumable(model_type, epochs=40, on_run_done=None):
    done_keys = load_completed_keys()
    cfgs = get_cfgs_for_type(model_type)

    total = len(cfgs)
    ran = 0
    skipped = 0

    print(f"{model_type}: {total} configs")

    for i, cfg in enumerate(cfgs, 1):
        key = make_config_key(model_type, cfg)

        if key in done_keys:
            skipped += 1
            continue

        ran += 1
        print(f"\n[{i}/{total}] RUN {model_type} cfg={cfg}")
        met = run_one(model_type, cfg, epochs)
        print("=>", met)

        if on_run_done is not None:
            on_run_done()

    print(f"{model_type} new runs={ran}, skipped={skipped}")


# In[ ]:


# ---- Grid search run script (fixed progress bar) ----

def make_overall_progress(total_remaining):
    bar = widgets.IntProgress(
        value=0, min=0, max=total_remaining,
        description="Gridsearch:",
        bar_style="",
        orientation="horizontal",
        layout=widgets.Layout(width="80%")
    )
    label = widgets.HTML(value=f"<b>0 / {total_remaining}</b> completed")
    sublabel = widgets.HTML(value="Waiting to start…")

    box = widgets.VBox([bar, label, sublabel])
    display(box)

    def update(n_done, text=""):
        bar.value = n_done
        label.value = f"<b>{n_done} / {total_remaining}</b> completed"
        if text:
            sublabel.value = text
    return update


def fmt_seconds(sec):
    sec = int(max(sec, 0))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0: return f"{h}h {m}m {s}s"
    if m > 0: return f"{m}m {s}s"
    return f"{s}s"


def get_last_model_type():
    if not LEADERBOARD_PATH.exists():
        return None
    df = pd.read_csv(LEADERBOARD_PATH, engine="python", on_bad_lines="skip")
    if len(df) == 0:
        return None
    return str(df.iloc[-1]["model_type"])


def rotate_model_types(model_types, start_type):
    if start_type is None or start_type not in model_types:
        return model_types
    i = model_types.index(start_type)
    return model_types[i:] + model_types[:i]


def run_all_searches_with_eta():
    # All families you want in the grid
    base_order = ["base_unet", "unet_fno", "att_unet", "att_unet_fno", "resnet1d"]

    # Freeze the set of already-completed configs at the time we start
    initial_done = load_completed_keys()

    # Compute how many configs will *actually* be run in THIS call
    per_family_remaining = {}
    total_to_run = 0
    for mt in base_order:
        cfgs = get_cfgs_for_type(mt)
        n = sum(
            1
            for cfg in cfgs
            if make_config_key(mt, cfg) not in initial_done
        )
        per_family_remaining[mt] = n
        total_to_run += n

    print("Total configs that will run in this call:", total_to_run)

    # If nothing is left, just show the leaderboard and exit
    if total_to_run == 0:
        print("Nothing left to run – all configs are already completed.")
        df = pd.read_csv(LEADERBOARD_PATH, engine="python", on_bad_lines="skip")
        display(df.sort_values("test_acc", ascending=False).head(10))
        print("\nBest per type:")
        display(df.sort_values("test_acc", ascending=False).groupby("model_type").head(1))
        return

    update_overall = make_overall_progress(total_to_run)

    # Rotate order so we resume from the family we last touched
    last_type = get_last_model_type()
    model_types = rotate_model_types(base_order, last_type)

    print("Last completed family:", last_type)
    print("Resume order:", model_types)

    done_so_far = 0
    start_time = time.time()

    for mt in model_types:
        rem_this = per_family_remaining.get(mt, 0)

        # If this family has nothing new to run, just update sublabel and skip
        if rem_this == 0:
            update_overall(
                done_so_far,
                f"<i>Skipping {mt} — no remaining configs in this family</i>"
            )
            continue

        update_overall(
            done_so_far,
            f"<i>Running {mt} — {rem_this} configs left in this family</i>"
        )

        def callback():
            nonlocal done_so_far
            done_so_far += 1

            elapsed = time.time() - start_time
            avg_per_run = elapsed / done_so_far if done_so_far > 0 else 0.0
            remaining = total_to_run - done_so_far
            eta_sec = remaining * avg_per_run

            update_overall(
                done_so_far,
                (
                    f"<i>Running {mt}…</i><br>"
                    f"Elapsed: <b>{fmt_seconds(elapsed)}</b> | "
                    f"Avg/run: <b>{fmt_seconds(avg_per_run)}</b> | "
                    f"ETA: <b>{fmt_seconds(eta_sec)}</b>"
                )
            )

        # This will only call `callback()` for truly new configs,
        # so `done_so_far` will end up == total_to_run.
        grid_search_resumable(
            mt,
            epochs=40,
            on_run_done=callback
        )

    elapsed_total = time.time() - start_time
    update_overall(
        done_so_far,
        f"<b>All searches complete ✅</b><br>Total time: <b>{fmt_seconds(elapsed_total)}</b>"
    )

    # Show leaderboard summary
    df = pd.read_csv(LEADERBOARD_PATH, engine="python", on_bad_lines="skip")
    display(df.sort_values("test_acc", ascending=False).head(10))

    print("\nBest per type:")
    display(df.sort_values("test_acc", ascending=False).groupby("model_type").head(1))


# Call this in the notebook to (re)run the gridsearch with a correct progress bar:
run_all_searches_with_eta()


# ## Results (so far)

# ### Stats for the runs

# In[17]:


# Evaluate ALL trained models (NO confusion matrices)

import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, balanced_accuracy_score
import tqdm.auto as tq

def eval_single_model_for_stats(model, test_loader, device="cuda"):
    """
    Evaluate a single model on the test set and compute:
    - accuracy
    - macro-F1 (over all classes)
    - balanced accuracy
    """
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    # Explicitly specify labels and zero_division for stability
    label_ids = np.arange(num_classes)

    acc      = float((y_true == y_pred).mean())
    macro_f1 = float(
        f1_score(
            y_true, y_pred,
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
    )
    bal_acc  = float(
        balanced_accuracy_score(
            y_true, y_pred
        )
    )

    return acc, macro_f1, bal_acc


def evaluate_all_models_stats():
    # IMPORTANT: dataset-scoped models directory
    MODELS_DIR = Path("models") / DATASET_NAME

    families = ["base_unet", "unet_fno", "att_unet", "att_unet_fno", "resnet1d"]
    rows = []

    # Count models robustly
    total_models = 0
    for f in families:
        fam_dir = MODELS_DIR / f
        if fam_dir.exists():
            total_models += len([d for d in fam_dir.glob("run_*") if d.is_dir()])

    print("Models directory:", MODELS_DIR.resolve())
    print("Found total trained models:", total_models)

    if total_models == 0:
        raise RuntimeError(
            f"No trained runs found under {MODELS_DIR}. "
            f"Expected structure like: models/{DATASET_NAME}/base_unet/run_XXX/"
        )

    pbar = tq.tqdm(total=total_models, desc=f"Evaluating models ({DATASET_NAME})")

    for family in families:
        fam_dir = MODELS_DIR / family
        if not fam_dir.exists():
            continue

        runs = sorted([d for d in fam_dir.glob("run_*") if d.is_dir()])

        for r in runs:
            config_path  = r / "config.json"
            metrics_path = r / "metrics.json"
            weights_path = r / "model.pt"

            if not (config_path.exists() and weights_path.exists()):
                pbar.update(1)
                continue

            # Load config
            with open(config_path, "r") as f:
                cfg = json.load(f)

            # Load model
            model = build_model(family, cfg)
            state = torch.load(weights_path, map_location=device)
            model.load_state_dict(state)
            model.to(device)

            # Evaluate on TEST set
            acc, macro_f1, bal_acc = eval_single_model_for_stats(
                model, test_loader, device
            )

            row = {
                "family": family,
                "run_id": r.name,
                **cfg,
                "test_acc": acc,
                "macro_f1": macro_f1,
                "balanced_acc": bal_acc,
            }

            # Add extra metrics from metrics.json
            if metrics_path.exists():
                with open(metrics_path, "r") as f:
                    saved_metrics = json.load(f)

                if "params_million" in saved_metrics:
                    row["params_million"] = saved_metrics["params_million"]
                if "val_acc" in saved_metrics:
                    row["val_acc"] = saved_metrics["val_acc"]

            rows.append(row)
            pbar.update(1)

    pbar.close()

    df = pd.DataFrame(rows)

    out_path = MODELS_DIR / "eval_stats.csv"
    df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    return df.sort_values("macro_f1", ascending=False)



df_stats = evaluate_all_models_stats()
df_stats


# In[18]:


# Create a CSV with top 5 models per family (dataset-scoped)

import pandas as pd
from pathlib import Path

def make_top5_per_family():
    MODELS_DIR = Path("models") / DATASET_NAME
    stats_path = MODELS_DIR / "eval_stats.csv"

    if not stats_path.exists():
        raise FileNotFoundError(f"eval_stats.csv not found at: {stats_path}")

    df = pd.read_csv(stats_path)

    families = ["base_unet", "unet_fno", "att_unet", "att_unet_fno", "resnet1d"]
    top_rows = []

    for fam in families:
        sub = df[df["family"] == fam]
        if sub.empty:
            print(f"Warning: No runs found for {fam}")
            continue

        top5 = sub.sort_values("macro_f1", ascending=False).head(5)
        top_rows.append(top5)

    if not top_rows:
        raise RuntimeError(f"No families had any models to select in {stats_path}")

    out_df = pd.concat(top_rows, ignore_index=True)

    out_path = MODELS_DIR / "top5_per_family.csv"
    out_df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    return out_df.sort_values(["family", "macro_f1"], ascending=[True, False])


df_top5 = make_top5_per_family()
df_top5


# In[ ]:


import pandas as pd
from pathlib import Path

df = pd.read_csv("models/eval_stats.csv")

# Example: pick the best ResNet1D run
best_resnet = df[df["family"] == "resnet1d"].sort_values("macro_f1", ascending=False).iloc[0]
p_ref = best_resnet["params_million"]

# Filter other families to those within ±10% of ResNet params
tol = 0.10  # 10%
mask = (df["params_million"] >= p_ref * (1 - tol)) & \
       (df["params_million"] <= p_ref * (1 + tol))

df_close = df[mask]

# From df_close, you can then pick the best per family
families = ["base_unet", "unet_fno", "att_unet", "att_unet_fno", "resnet1d"]
best_param_matched = []
for fam in families:
    sub = df_close[df_close["family"] == fam]
    if len(sub):
        best_param_matched.append(sub.sort_values("macro_f1", ascending=False).iloc[0])

best_param_matched = pd.DataFrame(best_param_matched)
best_param_matched


# ### Plots for evaluation

# In[27]:


eval_df = pd.read_csv("models/eval_stats.csv")
top5_df = pd.read_csv("models/top5_per_family.csv")


# In[ ]:


# Best Macro-F1 per family

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("models/eval_stats.csv")

families = ["base_unet", "unet_fno", "att_unet", "att_unet_fno", "resnet1d"]

best_rows = []
for fam in families:
    best = df[df["family"] == fam].sort_values("macro_f1", ascending=False).head(1)
    best_rows.append(best)

best_df = pd.concat(best_rows)

plt.figure(figsize=(8,5))
plt.bar(best_df["family"], best_df["macro_f1"], color="skyblue")
plt.ylabel("Macro-F1")
plt.title("Best Macro-F1 per Model Family")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.show()


# In[28]:


# Params vs Macro-F1

plt.figure(figsize=(8,6))

for fam in families:
    sub = df[df["family"] == fam]
    plt.scatter(sub["params_million"], sub["macro_f1"], label=fam, s=40)

plt.xlabel("Params (Millions)")
plt.ylabel("Macro-F1")
plt.title("Parameter Count vs Macro-F1")
plt.grid(alpha=0.3)
plt.legend()
plt.show()


# In[29]:


# Top-5 Macro-F1 Distribution per Family

import seaborn as sns

top5 = pd.read_csv("models/top5_per_family.csv")

plt.figure(figsize=(10,6))
sns.boxplot(data=top5, x="family", y="macro_f1", palette="Set2")
sns.stripplot(data=top5, x="family", y="macro_f1", color="black")
plt.title("Top-5 Macro-F1 per Model Family")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.25)
plt.show()


# In[30]:


# ===========================================================
# Export all evaluation figures to JPGs into figs/
#   - confusion_grid.jpg
#   - per_class_f1.jpg
#   - top5_boxplot.jpg
#   - params_vs_macroF1.jpg
# ===========================================================

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix, balanced_accuracy_score

# ---------- Paths & setup ----------
MODELS_DIR = Path("models")
EVAL_STATS_PATH = MODELS_DIR / "eval_stats.csv"
TOP5_PATH = MODELS_DIR / "top5_per_family.csv"
FIG_DIR = Path("figs") / DATASET_NAME
FIG_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

families = ["base_unet", "unet_fno", "att_unet", "att_unet_fno", "resnet1d"]
pretty_names = {
    "base_unet":     "Base U-Net",
    "unet_fno":      "U-Net + FNO",
    "att_unet":      "Attention U-Net",
    "att_unet_fno":  "Attention U-Net + FNO",
    "resnet1d":      "ResNet1D",
}

# ---------- Helper: build model from config ----------
def build_model_for_eval(family, cfg):
    """
    Build correct model instance for a family using saved config.
    Assumes these classes already exist in the notebook:
      - UNet1DClassifier
      - UNetFNO1DClassifier
      - AttUNet1DClassifier
      - AttUNetFNO1DClassifier
      - ResNet1DClassifier
    And global num_classes is defined.
    """
    base = int(cfg.get("base", 32))
    dropout = float(cfg.get("dropout", 0.0))
    modes = int(cfg.get("modes", 16)) if "modes" in cfg else 16
    fno_layers = int(cfg.get("fno_layers", 2)) if "fno_layers" in cfg else 2

    if family == "base_unet":
        return UNet1DClassifier(in_ch=1, base=base, num_classes=num_classes, dropout=dropout)
    elif family == "unet_fno":
        return UNetFNO1DClassifier(
            in_ch=1, base=base, num_classes=num_classes,
            modes=modes, fno_layers=fno_layers, dropout=dropout
        )
    elif family == "att_unet":
        return AttUNet1DClassifier(in_ch=1, base=base, num_classes=num_classes, dropout=dropout)
    elif family == "att_unet_fno":
        return AttUNetFNO1DClassifier(
            in_ch=1, base=base, num_classes=num_classes,
            modes=modes, fno_layers=fno_layers, dropout=dropout
        )
    elif family == "resnet1d":
        return ResNet1DClassifier(in_ch=1, base=base, num_classes=num_classes, dropout=dropout)
    else:
        raise ValueError(f"Unknown family: {family}")


# ---------- Helper: load best model per family ----------
def load_best_model(family, metric="macro_f1"):
    df = pd.read_csv(EVAL_STATS_PATH)
    sub = df[df["family"] == family]
    if sub.empty:
        raise ValueError(f"No rows found for family '{family}' in eval_stats.csv")

    best_row = sub.sort_values(metric, ascending=False).iloc[0]
    run_id = best_row["run_id"]
    run_dir = MODELS_DIR / family / run_id

    cfg_path = run_dir / "config.json"
    weights_path = run_dir / "model.pt"

    with open(cfg_path, "r") as f:
        cfg = json.load(f)

    model = build_model_for_eval(family, cfg)
    state = torch.load(weights_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    return model, best_row


# ---------- Helper: evaluate model (for confusion, stats) ----------
def evaluate_model_full(model, loader, classes):
    """
    Evaluate model on the given loader (test_loader in our use)
    and return confusion matrix + metrics in a dict.
    """
    model.eval()
    y_true_all, y_pred_all = [], []

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())

    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)

    label_ids = np.arange(len(classes))

    cm = confusion_matrix(y_true, y_pred, labels=label_ids)
    acc = float((y_true == y_pred).mean())
    macro_f1 = float(
        f1_score(
            y_true, y_pred,
            labels=label_ids,
            average="macro",
            zero_division=0,
        )
    )
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))

    return {
        "cm": cm,
        "acc": acc,
        "macro_f1": macro_f1,
        "balanced_acc": bal_acc,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ===========================================================
# 1) Confusion matrix grid  (now supports all 5 families)
# ===========================================================
print("Exporting confusion_grid.jpg .")

# Only keep families that actually exist in eval_stats
df_eval = pd.read_csv(EVAL_STATS_PATH)
fams_available = [f for f in families if (df_eval["family"] == f).any()]

n_fam = len(fams_available)
if n_fam == 0:
    print("No families found in eval_stats.csv, skipping confusion_grid.")
else:
    # Dynamic grid: up to 3 columns
    ncols = min(3, n_fam)
    nrows = int(np.ceil(n_fam / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(5.5 * ncols, 5.0 * nrows))
    axes = np.array(axes).reshape(-1)
    all_images = []

    for i, fam in enumerate(fams_available):
        ax = axes[i]
        model, best_row = load_best_model(fam, metric="macro_f1")
        res = evaluate_model_full(model, test_loader, classes)
        cm = res["cm"]

        im = ax.imshow(cm, cmap="Blues")
        all_images.append(im)

        ax.set_title(
            f"{pretty_names.get(fam, fam)}\nmacro-F1 = {res['macro_f1']:.3f}",
            fontsize=12, pad=10
        )
        ax.set_xticks(np.arange(len(classes)))
        ax.set_yticks(np.arange(len(classes)))
        ax.set_xticklabels(classes, rotation=45, ha="right")
        ax.set_yticklabels(classes)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")

        thresh = cm.max() * 0.6 if cm.max() > 0 else 0.0
        for (r, c), val in np.ndenumerate(cm):
            ax.text(
                c, r, str(val),
                ha="center", va="center",
                fontsize=9,
                color="white" if val > thresh else "black"
            )

    # Turn off any unused axes
    for j in range(n_fam, len(axes)):
        axes[j].axis("off")

    fig.subplots_adjust(
        left=0.08,
        right=0.88,
        top=0.90,
        bottom=0.15,
        wspace=0.35,
        hspace=0.35,
    )

    # Colorbar on the far right
    cbar_ax = fig.add_axes([0.90, 0.20, 0.02, 0.60])
    fig.colorbar(all_images[-1], cax=cbar_ax)

    fig.suptitle("Confusion Matrices — Best Model per Family", fontsize=16, y=0.98)

    plt.savefig(FIG_DIR / "confusion_grid.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ===========================================================
# 2) Per-class F1 comparison (includes ResNet if available)
# ===========================================================
print("Exporting per_class_f1.jpg .")

def compute_per_class_f1(model):
    model.eval()
    y_true_all, y_pred_all = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb).argmax(1)
            y_true_all.append(yb.cpu().numpy())
            y_pred_all.append(preds.cpu().numpy())
    y_true = np.concatenate(y_true_all)
    y_pred = np.concatenate(y_pred_all)
    return f1_score(y_true, y_pred, average=None)

per_class_f1_dict = {}
for fam in fams_available:
    try:
        model, _ = load_best_model(fam, metric="macro_f1")
    except Exception as e:
        print(f"Skipping {fam} in per_class_f1: {e}")
        continue
    per_class_f1_dict[fam] = compute_per_class_f1(model)

if per_class_f1_dict:
    x = np.arange(len(classes))
    width = 0.15

    fig, ax = plt.subplots(figsize=(11, 6))

    for i, fam in enumerate(per_class_f1_dict.keys()):
        ax.bar(x + i * width, per_class_f1_dict[fam], width, label=pretty_names.get(fam, fam))

    ax.set_xticks(x + width * (len(per_class_f1_dict) - 1) / 2)
    ax.set_xticklabels(classes, rotation=45)
    ax.set_ylabel("F1 Score")
    ax.set_title("Per-Class F1 Comparison Across Best Models")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIG_DIR / "per_class_f1.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
else:
    print("No models available to plot per-class F1.")


# ===========================================================
# 3) Top-5 per family boxplot
# ===========================================================
print("Exporting top5_boxplot.jpg .")

if TOP5_PATH.exists():
    top5 = pd.read_csv(TOP5_PATH)

    fig, ax = plt.subplots(figsize=(10, 6))
    top5.boxplot(column="macro_f1", by="family", ax=ax)
    ax.set_title("Top-5 Macro-F1 per Model Family")
    ax.set_ylabel("Macro F1")
    plt.suptitle("")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "top5_boxplot.jpg", dpi=300, bbox_inches="tight")
    plt.close(fig)
else:
    print("Warning: top5_per_family.csv not found, skipping top5_boxplot.jpg")


# ===========================================================
# 4) Params vs macro-F1 scatter
# ===========================================================
print("Exporting params_vs_macroF1.jpg .")

df_stats = pd.read_csv(EVAL_STATS_PATH)

fig, ax = plt.subplots(figsize=(9, 6))

for fam in families:
    sub = df_stats[df_stats["family"] == fam]
    if sub.empty:
        continue
    ax.scatter(
        sub["params_million"],
        sub["macro_f1"],
        label=pretty_names.get(fam, fam),
        alpha=0.7
    )

ax.set_xlabel("Params (Millions)")
ax.set_ylabel("Macro F1")
ax.set_title("Parameter Count vs Macro-F1")
ax.grid(alpha=0.3)
ax.legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "params_vs_macroF1.jpg", dpi=300, bbox_inches="tight")
plt.close(fig)

print("✅ All plots exported to 'figs/' as JPG files.")


# In[ ]:


import pandas as pd
from pathlib import Path

MODELS_DIR = Path("models")
eval_stats_path = MODELS_DIR / "eval_stats.csv"

df = pd.read_csv(eval_stats_path)

families = ["base_unet", "unet_fno", "att_unet", "att_unet_fno", "resnet1d"]
pretty = {
    "base_unet":    "U-Net",
    "unet_fno":     "U-Net + FNO",
    "att_unet":     "Attention U-Net",
    "att_unet_fno": "Attention U-Net + FNO",
    "resnet1d":     "ResNet1D",
}

print("% LaTeX rows for Table: Best configuration per model family")
for fam in families:
    sub = df[df["family"] == fam]
    if sub.empty:
        continue
    best = sub.sort_values("macro_f1", ascending=False).iloc[0]

    family_name = pretty[fam]
    macro  = best["macro_f1"]
    bal    = best["balanced_acc"]
    acc    = best["test_acc"]
    params = best.get("params_million", float("nan"))
    run_id = best["run_id"]
    base   = best["base"]

    row = (
        f"{family_name:23} & "
        f"{macro:.3f} & {bal:.3f} & {acc:.3f} & "
        f"{params:.2f} & \\texttt{{{run_id}}} & {int(base)} \\\\"
    )
    print(row)


# In[19]:


get_ipython().system('python -m nbconvert --to script Pipeline.ipynb')


# In[ ]:




