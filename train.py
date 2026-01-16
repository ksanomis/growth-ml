import torch

def train_model(model, X, y, lr, epochs):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_hist = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        loss_hist.append(loss.item())

    return loss_hist, pred.detach().numpy()
