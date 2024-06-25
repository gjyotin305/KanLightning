from KanBERT.kan_bert_tokenizer import KanTokenizer
from dataset import IMDBDataset, load_imdb_data
from torch.utils.data import DataLoader
from torch.utils.data import random_split

text, labels = load_imdb_data("./data/IMDB Dataset.csv")

tokenizer = KanTokenizer()
tokenizer.ingest_vocab_batch(text=text)

dataset_data = IMDBDataset(text=text, labels=labels, tokenizer=tokenizer)

lengths = [int(len(dataset_data))*0.8, int(len(dataset_data)*0.2)]
train_dataset, val_dataset = random_split(dataset=dataset_data, lengths=lengths)

train_dataloader = DataLoader(dataset=train_dataset, num_workers=8, shuffle=True, batch_size=4)
val_dataloader = DataLoader(dataset=val_dataset, num_workers=8, shuffle=True, batch_size=4)


