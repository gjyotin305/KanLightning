from typing import (Any, List,
                    Dict)
from KanBERT.kan_bert_tokenizer import KanTokenizer
from KanBERT.kan_bert_utils import accuracy
from KanBERT.kan_bert import KanBert
import torch.nn as nn
import torch
import wandb
from KanBERT.constants import device, maxlen
from einops import rearrange
from dataset import IMDBDataset, load_imdb_data
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from lightning.pytorch.loggers import WandbLogger
import lightning as pl
from torch.utils.data import random_split

text, labels = load_imdb_data("./data/IMDB Dataset.csv")


tokenizer = KanTokenizer(max_length=maxlen)
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
        self.train_acc = Accuracy(task="binary")
        self.val_acc = Accuracy(task="binary")
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits_clf = self.model.forward(x[0].to(device), x[1].to(device))
        loss = self.loss(logits_clf, y.to(device).float())
        pred_logits = self.sigmoid(logits_clf)

        self.log_dict({"train_loss":loss}, 
                 on_epoch=True, 
                 on_step=True, 
                 prog_bar=True)
        
        _, preds = torch.max(pred_logits, dim=1)
        label = rearrange(y, "b h -> b")
        self.train_acc(preds, y)

        return loss
    
    def on_train_epoch_end(self) -> None:
        train_acc_end = self.train_acc.compute()
        self.log_dict({"train_acc_end": train_acc_end})
        self.train_acc.reset()


    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        
        logits_clf = self.model.forward(x[0].to(device), x[1].to(device))
        
        pred_logits = self.sigmoid(logits_clf)

        val_acc = accuracy(pred_logits, labels=y.to(device))
        _, preds = torch.max(pred_logits, dim=1)
        label = rearrange(y, "b h -> b")
        self.val_acc(preds, label)
    
    def on_validation_epoch_end(self) -> None:
        val_acc_end = self.val_acc.compute()
        self.log_dict({"val_acc_end": val_acc_end})
        self.val_acc.reset()
        

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-5)
        return optimizer
    
wandb_logger = WandbLogger(project="kanert")
model_light = KanBertLightning()
trainer = pl.Trainer(accelerator="gpu", 
                     max_epochs=100, 
                     logger=wandb_logger, 
                     check_val_every_n_epoch=5)
trainer.fit(model=model_light, 
            train_dataloaders=train_dataloader, 
            val_dataloaders=val_dataloader)

                




