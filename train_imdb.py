from KanBERT.kan_bert_tokenizer import KanTokenizer
from dataset import IMDBDataset, load_imdb_data

text, labels = load_imdb_data("./data/IMDB Dataset.csv")

tokenizer = KanTokenizer()
tokenizer.ingest_vocab_batch(text=text)

dataset_train = IMDBDataset(text=text, labels=labels, tokenizer=tokenizer)

for i in dataset_train:
    print(i)
    break