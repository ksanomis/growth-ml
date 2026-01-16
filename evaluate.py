import matplotlib.pyplot as plt
import numpy as np

def plot_true_vs_pred(y_true, y_pred, title):
    plt.figure(figsize=(5,4))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot([0,100], [0,100], '--', color='black')
    plt.xlabel("True coverage (%)")
    plt.ylabel("Predicted coverage (%)")
    plt.title(title)
    plt.grid(True)
    plt.show()


def plot_loss(loss_hist, title):
    plt.figure(figsize=(5,4))
    plt.plot(loss_hist)
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(title)
    plt.yscale("log")
    plt.grid(True)
    plt.show()
