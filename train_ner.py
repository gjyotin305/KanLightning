import torch
import lightning as pl
import torch.nn as nn
from einops import rearrange
from KANFormer.KANFormer import KANFormer
import torch.nn.functional as F
from torch.masked import masked_tensor
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from NER.input_pipeline import NER_ConLL, ner_id_to_tok, ner_tok_to_id, vocab_check


dataset_train = NER_ConLL(mode="train", vocab=vocab_check,maxlen=128)
dataset_test = NER_ConLL(mode="test", vocab=vocab_check,maxlen=128)

def lossfn(pred_label, target_label):
    loss = (-pred_label.log() * target_label).sum(dim=1)
    return loss

train_dataloader = DataLoader(dataset=dataset_train, batch_size=16, num_workers=15, shuffle=True)
test_dataloader = DataLoader(dataset=dataset_test, batch_size=16, num_workers=15, shuffle=True)

model = KANFormer(vocabulary_size=len(vocab_check), hidden_size=32, num_heads=2, window_size=4, d_ff=64, num_experts=4, n_experts_per_token=2, n_blocks=6, max_seq_len=128, num_tags_ner=9)

class KANLightning(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.kan = model

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.kan(x)
        out = torch.softmax(logits, dim=2)
        print(out.shape)
        labels = out.argmax(dim=2)
        loss = lossfn(labels, y)
        print(loss.shape)
        mask = y<9# To not include loss of padded tokens
        mt = masked_tensor(y.float(), mask)
        loss = loss * mt
        final_loss = torch.sum(loss) / torch.sum(mt)
        self.log("train_loss", final_loss)
        return final_loss


    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.kan(x)
        loss = lossfn(out, y)
        mask = loss<9 # To not include loss of padded tokens
        mt = masked_tensor(loss.float(), mask)
        loss = torch.dot(loss, mt)
        accuracy = (out.argmax(dim=1) == y).float().mean()
        self.log("test_acc", accuracy)
        self.log("test_loss", loss)
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)
        return [optimizer], [scheduler]

model_ll = KANLightning()

trainer = pl.Trainer(max_epochs=1)
trainer.fit(train_dataloaders=train_dataloader, model=model_ll)