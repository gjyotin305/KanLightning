import argparse
import torch
import lightning as pl
import torch.nn as nn
from einops import rearrange
from model import KAN
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

lossfn = nn.CrossEntropyLoss()

device = "cuda" if torch.cuda.is_available() else "cpu"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,),(0.3081))
])
dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)

train_loader = DataLoader(dataset1, batch_size=32, num_workers=15)
test_loader = DataLoader(dataset2, batch_size=32, num_workers=15)

print(dataset1.__len__())

class KANLightning(pl.LightningModule):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.kan = KAN([input_size, hidden_size, output_size])

    def training_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, "a b c d -> a (b c d)")
        out = self.kan(x)
        loss = lossfn(out, y)
        self.log("train_loss", loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        x = rearrange(x, "a b c d -> a (b c d)")
        out = self.kan(x)
        loss = lossfn(out, y)
        accuracy = (out.argmax(dim=1) == y).float().mean()
        self.log("test_acc", accuracy)
        self.log("test_loss", loss)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        return [optimizer], [scheduler]

model = KANLightning(28*28, 32, 10)
trainer = pl.Trainer(max_epochs=1)

trainer.fit(model=model, train_dataloaders=train_loader)
trainer.test(dataloaders=test_loader) 

