import numpy as np
import pandas as pd

def generate_langmuir_data(phys_cfg, data_cfg):
    flow_A = np.random.uniform(0.5, 10.0, data_cfg.n_samples)
    flow_B = np.random.uniform(0.5, 10.0, data_cfg.n_samples)
    T = np.random.uniform(data_cfg.temp_min, data_cfg.temp_max, data_cfg.n_samples)
    t = np.random.uniform(data_cfg.time_min, data_cfg.time_max, data_cfg.n_samples)

    k_eff = phys_cfg.k0 * flow_A * flow_B * np.exp(-phys_cfg.Ea / (phys_cfg.R * T))
    theta = phys_cfg.theta_inf * (1 - np.exp(-k_eff * t))

    noise = np.random.normal(0, data_cfg.noise_level, len(theta))
    theta = np.clip(theta * (1 + noise), 0, 1)

    return pd.DataFrame({
        "growth_time": t,
        "precursor_A": flow_A,
        "precursor_B": flow_B,
        "temperature": T,
        "coverage": theta * 100
    })


def scale_features(X):
    return (X - X.mean(axis=0)) / X.std(axis=0)
