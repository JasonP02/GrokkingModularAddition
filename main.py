# imports
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class OneLayerTransformer(nn.Module):
    def __init__(
            self,
            d_model = 128,
            n_head = 4,
            num_encoder_layers = 1,
            num_decoder_layers = 1,
            dim_feedforward = 512,
            dropout = 0.5,
            activation = "relu"):
        pass

network = OneLayerTransformer()

class AdditionDataset(Dataset):
    def __init__(self, P):
        self.P = P
        self.data = [[a, b, a + b] for a in range(self.P) for b in range(self.P)]

    def __len__(self):
        return self.P**2

    def __getitem__(self, idx):
        return self.data[idx][:2], self.data[idx][2]

# Boilerplate training loop
if __name__ == "__main__":
    P = 113
    batch_size = 32
    epochs = 2
    lr = 1e-3

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = AdditionDataset(P)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )

    model = OneLayerTransformer().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for xb, yb in dataloader:
            # Move to GPU
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            # Dummy forward, replace with actual model logic
            logits = torch.randn(xb.shape[0], P, device=device)  # [batch, num_classes]
            loss = criterion(logits, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

