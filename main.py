import numpy as np
import torch

from config import *
from data import *
from models import *
from train import *
from evaluate import *

phys_cfg = PhysicalConfig()
data_cfg = DataConfig()
train_cfg = TrainConfig()
cnn_cfg = CNNConfig()


np.random.seed(train_cfg.seed)
torch.manual_seed(train_cfg.seed)

df = generate_langmuir_data(phys_cfg, data_cfg)

X = df[['growth_time','precursor_A','precursor_B','temperature']].values
y = df[['coverage']].values

X_scaled = scale_features(X)

X_t = torch.tensor(X_scaled, dtype=torch.float32)
y_t = torch.tensor(y, dtype=torch.float32)

# ---- Linear baseline ----
lin_model = LinearModel(X_t.shape[1])
loss_lin, y_pred_lin = train_model(
    lin_model, X_t, y_t,
    train_cfg.lr_linear,
    train_cfg.epochs_linear
)

plot_true_vs_pred(y, y_pred_lin, "Linear Regression")

# ---- Neural network ----
nn_model = CoverageNN(X_t.shape[1])
loss_nn, y_pred_nn = train_model(
    nn_model, X_t, y_t,
    train_cfg.lr_nn,
    train_cfg.epochs_nn
)

plot_true_vs_pred(y, y_pred_nn, "Neural Network")
