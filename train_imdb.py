from typing import (Any, List,
                    Dict)
from KanBERT.kan_bert_tokenizer import KanTokenizer
from KanBERT.kan_bert_utils import accuracy
from KanBERT.kan_bert import KanBert
import torch.nn as nn
import torch
import wandb
from KanBERT.constants import device
from dataset import IMDBDataset, load_imdb_data
from torch.utils.data import DataLoader
from lightning.pytorch.loggers import WandbLogger
import lightning as pl
from torch.utils.data import random_split

text, labels = load_imdb_data("./data/IMDB Dataset.csv")


tokenizer = KanTokenizer()
tokenizer.ingest_vocab_batch(text=text)

dataset_data = IMDBDataset(text=text, labels=labels, tokenizer=tokenizer)

print(tokenizer.vocab_size)
model = KanBert(vocab_size=len(tokenizer.vocab_dict))

lengths = [int(len(dataset_data)*0.8), int(len(dataset_data)*0.2)]
train_dataset, val_dataset = random_split(dataset=dataset_data, lengths=lengths)

train_dataloader = DataLoader(dataset=train_dataset, 
                            shuffle=True, 
                            batch_size=4)

val_dataloader = DataLoader(dataset=val_dataset, 
                            shuffle=True, 
                            batch_size=4)
class KanBertLightning(pl.LightningModule):
    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits_clf = self.model.forward(x[0].to(device), x[1].to(device))
        loss = self.loss(logits_clf, y.to(device))
        pred_logits = self.sigmoid(logits_clf)

        self.log({"train_loss":loss}, 
                 on_epoch=True, 
                 on_step=True, 
                 prog_bar=True)

        train_acc = accuracy(pred_logits, y)

        self.log({"train_acc": train_acc},
                on_epoch=True,
                prog_bar=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        
        logits_clf = self.model.forward(x[0].to(device), x[1].to(device))
        
        pred_logits = self.sigmoid(logits_clf)
        val_acc = accuracy(pred_logits, labels=y.to(device))
        self.log({"val_acc": val_acc}, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
    
wandb_logger = WandbLogger(project="kanert")
model_light = KanBertLightning()
trainer = pl.Trainer(accelerator="gpu", max_epochs=4, logger=wandb_logger)
trainer.fit(model=model_light, 
            train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

                




