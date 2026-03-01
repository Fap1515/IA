# minitorch_felipe.py
from __future__ import annotations

from typing import Dict, List, Tuple, Optional, Callable
import torch


# =========================
# Core container
# =========================
class Net:
    """
    A simple sequential container for custom layers.

    Provides train()/eval() switches and runs forward/backward/update
    across all layers.
    """

    def __init__(self):
        self.layers: List[object] = []
        self.training: bool = True

    def add(self, layer) -> None:
        """Add a layer to the network."""
        self.layers.append(layer)

    # ---- Mode control ----
    def train(self):
        """Switch the whole network to training mode."""
        self.training = True
        for layer in self.layers:
            if hasattr(layer, "train"):
                layer.train()
        return self

    def eval(self):
        """Switch the whole network to evaluation mode."""
        self.training = False
        for layer in self.layers:
            if hasattr(layer, "eval"):
                layer.eval()
        return self

    # ---- Core passes ----
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """Forward pass through all layers."""
        for layer in self.layers:
            X = layer.forward(X)
        return X

    def backward(self, dZ: torch.Tensor) -> torch.Tensor:
        """Backward pass through all layers in reverse order."""
        for layer in reversed(self.layers):
            dZ = layer.backward(dZ)
        return dZ

    def update(self, lr: float) -> None:
        """SGD update for all trainable layers (v1 training loop)."""
        for layer in self.layers:
            if hasattr(layer, "update"):
                layer.update(lr)


# =========================
# Layers
# =========================
class Linear:
    """
    Fully connected layer: Z = XW + b
    """

    def __init__(self, nin: int, nout: int, device: str = "cpu", weight_scale: float = 0.01):
        # Small init helps stability (especially for deeper nets)
        self.W = (weight_scale * torch.randn(nin, nout, device=device, requires_grad=False))
        self.b = torch.zeros(nout, device=device, requires_grad=False)

        self.training = True
        self.X: Optional[torch.Tensor] = None
        self.dW: Optional[torch.Tensor] = None
        self.db: Optional[torch.Tensor] = None

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.X = X
        return X @ self.W + self.b

    def backward(self, dZ: torch.Tensor) -> torch.Tensor:
        self.db = dZ.sum(dim=0)
        self.dW = self.X.t() @ dZ
        dX = dZ @ self.W.t()
        return dX

    def update(self, lr: float):
        self.W -= lr * self.dW
        self.b -= lr * self.db


class ReLU:
    """ReLU activation."""

    def __init__(self):
        self.Z: Optional[torch.Tensor] = None

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        self.Z = Z
        return torch.maximum(Z, torch.zeros_like(Z))

    def backward(self, dA: torch.Tensor) -> torch.Tensor:
        return dA * (self.Z > 0)

    def update(self, lr: float):
        # no params
        pass


class LeakyReLU:
    """Leaky ReLU activation."""

    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.Z: Optional[torch.Tensor] = None

    def forward(self, Z: torch.Tensor) -> torch.Tensor:
        self.Z = Z
        return torch.maximum(Z, self.alpha * Z)

    def backward(self, dA: torch.Tensor) -> torch.Tensor:
        dZ = torch.where(self.Z > 0, 1.0, self.alpha)
        return dA * dZ

    def update(self, lr: float):
        # no params
        pass


class BatchNorm1D:
    """
    Batch Normalization for 2D inputs: (batch, features).

    TRAIN: compute batch stats, normalize, update running stats, support backward().
    EVAL:  use running stats.
    """

    def __init__(self, n_features: int, eps: float = 1e-5, momentum: float = 0.1, device: str = "cpu"):
        self.eps = eps
        self.momentum = momentum
        self.device = device

        self.gamma = torch.ones(n_features, device=device, requires_grad=False)
        self.beta  = torch.zeros(n_features, device=device, requires_grad=False)

        self.running_mean = torch.zeros(n_features, device=device, requires_grad=False)
        self.running_var  = torch.ones(n_features,  device=device, requires_grad=False)

        self.training = True

        # caches
        self.X: Optional[torch.Tensor] = None
        self.X_hat: Optional[torch.Tensor] = None
        self.batch_mean: Optional[torch.Tensor] = None
        self.batch_var: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None

        # grads
        self.dgamma: Optional[torch.Tensor] = None
        self.dbeta: Optional[torch.Tensor] = None

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        self.X = X

        if self.training:
            self.batch_mean = X.mean(dim=0)
            self.batch_var  = X.var(dim=0, unbiased=False)
            self.std = torch.sqrt(self.batch_var + self.eps)
            self.X_hat = (X - self.batch_mean) / self.std

            # EMA update
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * self.batch_mean
            self.running_var  = (1 - self.momentum) * self.running_var  + self.momentum * self.batch_var
        else:
            std = torch.sqrt(self.running_var + self.eps)
            self.X_hat = (X - self.running_mean) / std

        return self.gamma * self.X_hat + self.beta

    def backward(self, dY: torch.Tensor) -> torch.Tensor:
        if not self.training:
            raise RuntimeError("Backward called in eval() mode. Use training mode for gradient computation.")

        m = dY.size(0)

        # parameter grads
        self.dbeta = dY.sum(dim=0)
        self.dgamma = (dY * self.X_hat).sum(dim=0)

        # input grads
        dx_hat = dY * self.gamma
        x_mu = self.X - self.batch_mean
        invstd = 1.0 / self.std

        dvar = torch.sum(dx_hat * x_mu * -0.5 * (invstd ** 3), dim=0)
        dmean = torch.sum(-dx_hat * invstd, dim=0) + dvar * torch.mean(-2.0 * x_mu, dim=0)

        dX = dx_hat * invstd + (2.0 / m) * x_mu * dvar + dmean / m
        return dX

    def update(self, lr: float):
        self.gamma -= lr * self.dgamma
        self.beta  -= lr * self.dbeta


class Dropout:
    """
    Inverted Dropout for tensors (batch, features).

    TRAIN: mask ~ Bernoulli(keep_prob), output = X * mask/keep_prob
    EVAL:  identity
    """

    def __init__(self, p: float = 0.5, device: str = "cpu"):
        assert 0.0 <= p < 1.0, "p must be in [0, 1)."
        self.p = p
        self.device = device
        self.training = True
        self.mask: Optional[torch.Tensor] = None

    def train(self):
        self.training = True
        return self

    def eval(self):
        self.training = False
        return self

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0:
            keep_prob = 1.0 - self.p
            self.mask = (torch.rand_like(X) < keep_prob).float()
            self.mask = self.mask / keep_prob
            return X * self.mask
        else:
            self.mask = None
            return X

    def backward(self, dY: torch.Tensor) -> torch.Tensor:
        if self.training and self.p > 0.0:
            return dY * self.mask
        return dY

    def update(self, lr: float):
        pass


# =========================
# Loss
# =========================
class CrossEntropyFromLogits:
    """
    Softmax + Cross-Entropy from raw logits.

    forward(Z, Y) returns scalar loss.
    backward(n_classes) returns dZ = (softmax(Z) - one_hot(Y)) / batch
    """

    def __init__(self):
        self.Y: Optional[torch.Tensor] = None
        self.A: Optional[torch.Tensor] = None  # softmax probs

    def forward(self, Z: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        self.Y = Y

        Z_stable = Z - Z.max(dim=1, keepdim=True).values
        expZ = torch.exp(Z_stable)
        self.A = expZ / expZ.sum(dim=1, keepdim=True)

        log_softmax_Z = Z_stable - torch.log(expZ.sum(dim=1, keepdim=True))

        batch_size = Z.size(0)
        log_probs = log_softmax_Z[torch.arange(batch_size, device=Z.device), self.Y]
        loss = -log_probs.mean()
        return loss

    def backward(self, n_classes: int) -> torch.Tensor:
        m = self.Y.size(0)
        Y_one_hot = torch.zeros(m, n_classes, device=self.A.device)
        Y_one_hot[torch.arange(m, device=self.A.device), self.Y] = 1.0
        dZ = (self.A - Y_one_hot) / m
        return dZ


# =========================
# Architectures
# =========================
def build_mnist_net(
    arch: str,
    *,
    device: str = "cpu",
    p_drop: float = 0.5,
    activation: str = "relu",   # "relu" | "leaky"
    leaky_alpha: float = 0.01,
) -> Net:
    """
    Build MNIST architectures used in the assignment.

    arch:
      - "baseline": Linear(784 -> 10)
      - "A":        Linear(784 -> 256) -> ReLU -> Linear(256 -> 10)
      - "B":        Linear(784 -> 256) -> BN -> ReLU -> Dropout -> Linear(256 -> 10)
      - "C":        784->512->BN->ReLU->Dropout->256->BN->ReLU->Dropout->10
    """
    net = Net()

    def act():
        if activation.lower() == "leaky":
            return LeakyReLU(alpha=leaky_alpha)
        return ReLU()

    arch = arch.lower()
    if arch in ("baseline", "softmax", "logreg"):
        net.add(Linear(784, 10, device=device))
        return net

    if arch == "a":
        net.add(Linear(784, 256, device=device))
        net.add(act())
        net.add(Linear(256, 10, device=device))
        return net

    if arch == "b":
        net.add(Linear(784, 256, device=device))
        net.add(BatchNorm1D(256, device=device))  # BN before activation
        net.add(act())
        net.add(Dropout(p=p_drop, device=device))
        net.add(Linear(256, 10, device=device))
        return net

    if arch == "c":
        net.add(Linear(784, 512, device=device))
        net.add(BatchNorm1D(512, device=device))
        net.add(act())
        net.add(Dropout(p=p_drop, device=device))

        net.add(Linear(512, 256, device=device))
        net.add(BatchNorm1D(256, device=device))
        net.add(act())
        net.add(Dropout(p=p_drop, device=device))

        net.add(Linear(256, 10, device=device))
        return net

    if arch == "c":
        net.add(Linear(784, 512, device=device))
        net.add(BatchNorm1D(512, device=device))
        net.add(act())
        net.add(Dropout(p=p_drop, device=device))

        net.add(Linear(512, 256, device=device))
        net.add(BatchNorm1D(256, device=device))
        net.add(act())
        net.add(Dropout(p=p_drop, device=device))

        net.add(Linear(256, 10, device=device))
        return net
    raise ValueError(f"Unknown architecture: {arch}")


# =========================
# Training utilities (ORIGINALS: keep as-is)
# =========================
def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(
    net: Net,
    loader,
    loss_fn: CrossEntropyFromLogits,
    device: str = "cpu",
    n_classes: int = 10
) -> Tuple[float, float]:
    net.eval()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0

    for X, y in loader:
        X = X.to(device).view(X.size(0), -1)
        y = y.to(device)

        logits = net.forward(X)
        loss = loss_fn.forward(logits, y)

        total_loss += float(loss.item())
        total_acc  += accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches


def train_one_epoch(
    net: Net,
    loader,
    loss_fn: CrossEntropyFromLogits,
    lr: float,
    device: str = "cpu",
    n_classes: int = 10
) -> Tuple[float, float, List[float]]:
    net.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    batch_losses: List[float] = []

    for X, y in loader:
        X = X.to(device).view(X.size(0), -1)
        y = y.to(device)

        logits = net.forward(X)
        loss = loss_fn.forward(logits, y)

        dZ = loss_fn.backward(n_classes=n_classes)
        net.backward(dZ)
        net.update(lr)

        total_loss += float(loss.item())
        total_acc  += accuracy_from_logits(logits, y)
        n_batches += 1
        batch_losses.append(float(loss.item()))

    return total_loss / n_batches, total_acc / n_batches, batch_losses


def train_model(
    net: Net,
    trainloader,
    valloader,
    epochs: int = 8,
    lr: float = 0.1,
    device: str = "cpu",
    n_classes: int = 10
) -> Tuple[Dict[str, List[float]], List[float]]:
    loss_fn = CrossEntropyFromLogits()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    all_batch_losses: List[float] = []

    for ep in range(1, epochs + 1):
        tr_loss, tr_acc, batch_losses = train_one_epoch(
            net, trainloader, loss_fn, lr, device=device, n_classes=n_classes
        )
        va_loss, va_acc = evaluate(
            net, valloader, loss_fn, device=device, n_classes=n_classes
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        all_batch_losses.extend(batch_losses)

        print(
            f"Epoch {ep:02d} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

    return history, all_batch_losses


# =========================
# Optimizer + schedules (NEW)
# =========================
class SGDMomentum:
    """
    Minimal SGD with momentum + L2 (weight decay).

    Notes:
      - weight_decay is applied ONLY to Linear.W (common practice).
      - no weight decay on Linear.b.
      - This version is intentionally minimal and matches your class.
    """

    def __init__(self, lr=0.1, momentum=0.9, weight_decay=1e-4):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v = {}  # key -> velocity tensor

    def _vel(self, key, param):
        if key not in self.v:
            self.v[key] = torch.zeros_like(param)
        return self.v[key]

    def step(self, net: Net):
        for layer in net.layers:
            if isinstance(layer, Linear):

                # ---- W (with L2) ----
                if layer.dW is not None:
                    gW = layer.dW + self.weight_decay * layer.W
                    vW = self._vel((id(layer), "W"), layer.W)
                    vW.mul_(self.momentum).add_(gW)
                    layer.W.add_(-self.lr * vW)

                # ---- b (no L2) ----
                if layer.db is not None:
                    gb = layer.db
                    vb = self._vel((id(layer), "b"), layer.b)
                    vb.mul_(self.momentum).add_(gb)
                    layer.b.add_(-self.lr * vb)


def step_lr(ep: int, base_lr: float, step_size: int = 40, gamma: float = 0.1) -> float:
    """
    Step decay schedule.

    Every 'step_size' epochs:
      lr = lr * gamma

    Example: base_lr=0.1, step_size=40, gamma=0.1
      ep 1-40   -> 0.1
      ep 41-80  -> 0.01
      ep 81-120 -> 0.001
    """
    k = (ep - 1) // step_size
    return base_lr * (gamma ** k)


# =========================
# Training utilities v2 (NEW) — with optional optimizer + optional schedule
# =========================
def train_one_epoch_opt(
    net: Net,
    loader,
    loss_fn: CrossEntropyFromLogits,
    lr: float,
    device: str = "cpu",
    n_classes: int = 10,
    opt: Optional[object] = None,
) -> Tuple[float, float, List[float]]:
    """
    Same as train_one_epoch, but if 'opt' is provided, uses:
        opt.lr = lr; opt.step(net)
    instead of net.update(lr).
    """
    net.train()
    total_loss, total_acc, n_batches = 0.0, 0.0, 0
    batch_losses: List[float] = []

    for X, y in loader:
        X = X.to(device).view(X.size(0), -1)
        y = y.to(device)

        logits = net.forward(X)
        loss = loss_fn.forward(logits, y)

        dZ = loss_fn.backward(n_classes=n_classes)
        net.backward(dZ)

        if opt is None:
            net.update(lr)
        else:
            opt.lr = lr
            opt.step(net)

        total_loss += float(loss.item())
        total_acc  += accuracy_from_logits(logits, y)
        n_batches += 1
        batch_losses.append(float(loss.item()))

    return total_loss / n_batches, total_acc / n_batches, batch_losses


def train_model_opt(
    net: Net,
    trainloader,
    valloader,
    epochs: int = 8,
    lr: float = 0.1,
    device: str = "cpu",
    n_classes: int = 10,
    opt: Optional[object] = None,
    lr_schedule: Optional[Callable[[int, float], float]] = None,
) -> Tuple[Dict[str, List[float]], List[float]]:
    """
    Train loop v2.

    - If opt is None -> uses net.update(lr_ep) (classic SGD)
    - If opt is provided -> uses opt.step(net) (e.g., SGDMomentum)

    lr_schedule:
      A function that takes (epoch, base_lr) and returns lr for that epoch.
      Example: lr_schedule=lambda ep, base_lr: step_lr(ep, base_lr, step_size=40, gamma=0.1)
    """
    loss_fn = CrossEntropyFromLogits()

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    all_batch_losses: List[float] = []

    for ep in range(1, epochs + 1):
        lr_ep = lr_schedule(ep, lr) if lr_schedule else lr

        tr_loss, tr_acc, batch_losses = train_one_epoch_opt(
            net, trainloader, loss_fn, lr_ep, device=device, n_classes=n_classes, opt=opt
        )
        va_loss, va_acc = evaluate(
            net, valloader, loss_fn, device=device, n_classes=n_classes
        )

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        all_batch_losses.extend(batch_losses)

        print(
            f"Epoch {ep:02d} | lr {lr_ep:.6f} | "
            f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
            f"val loss {va_loss:.4f} acc {va_acc:.4f}"
        )

    return history, all_batch_losses

# =========================
# Plotting utilities
# =========================
def plot_per_batch_loss(batch_losses: List[float], title: str = "Per-batch loss"):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.plot(range(1, len(batch_losses) + 1), batch_losses)
    plt.xlabel("batch step")
    plt.ylabel("loss")
    plt.title(title)
    plt.show()


def plot_history(history: Dict[str, List[float]], title: str = ""):
    import matplotlib.pyplot as plt

    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure()
    plt.plot(epochs, history["train_loss"], label="train")
    plt.plot(epochs, history["val_loss"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title(f"Loss {title}")
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(epochs, history["train_acc"], label="train")
    plt.plot(epochs, history["val_acc"], label="val")
    plt.xlabel("epoch")
    plt.ylabel("accuracy")
    plt.title(f"Accuracy {title}")
    plt.legend()
    plt.show()

import torch
import matplotlib.pyplot as plt
import pandas as pd

def set_seed(seed=123):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

device = "cuda" if torch.cuda.is_available() else "cpu"
set_seed(123)
device
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import random_split, DataLoader

