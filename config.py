from dataclasses import dataclass

@dataclass
class PhysicalConfig:
    R: float = 8.314
    Ea: float = 20000
    k0: float = 0.1
    theta_inf: float = 1.0

@dataclass
class DataConfig:
    n_samples: int = 100
    noise_level: float = 0.05
    time_min: float = 1.0
    time_max: float = 20.0
    temp_min: float = 650 + 273.15
    temp_max: float = 750 + 273.15

@dataclass
class TrainConfig:
    seed: int = 42
    lr_linear: float = 0.01
    lr_nn: float = 0.01
    epochs_linear: int = 1000
    epochs_nn: int = 500
    batch_size: int = 8

@dataclass
class CNNConfig:
    n_timepoints: int = 50
    lr: float = 1e-3
    epochs: int = 50
