from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchtext.vocab import vocab
from tqdm import tqdm
import numpy as np
import torch
from collections import Counter, OrderedDict


dataset = load_dataset("eriktks/conll2003")

ner_tok_to_id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8, "<pad>": 9}
ner_id_to_tok = {y:x for x, y in ner_tok_to_id.items()}

def build_vocab(dataset, vocab_size=20000):
    all_tokens = sum(dataset["train"]["tokens"], [])
    all_tokens_array = np.array(list(map(str.lower, all_tokens)))
    counter = Counter(all_tokens_array)
    vocab_ll = [token for token, count in counter.most_common(vocab_size - 2)]
    vocab_trans = vocab(OrderedDict([(token, 1) for token in vocab_ll]), special_first=True, specials=["<pad>", "<unk>"])
    vocab_trans.set_default_index(vocab_trans["<unk>"])
    return vocab_trans


vocab_check = build_vocab(dataset=dataset)

class NER_ConLL(Dataset):
    def __init__(self, mode, vocab ,maxlen=128) -> None:
        super().__init__()
        self.dataset = dataset
        self.split = mode
        self.vocab = vocab 
        self.__splitselect = self.dataset[self.split]
        self.maxlen = maxlen

    def __len__(self):
        return len(self.dataset[self.split])
    
    def __getitem__(self, idx):
        sample_data = self.__splitselect[idx]

        sample_data_x = sample_data["tokens"]
        sample_data_y = sample_data["ner_tags"]
        
        sample_data_x.extend(["<pad>"]*(self.maxlen - len(sample_data_x)))
        sample_data_y.extend([9]*(self.maxlen - len(sample_data_y)))

        assert len(sample_data_y) == self.maxlen
        assert len(sample_data_x) == self.maxlen

        torch_data_x = self.vocab(sample_data_x) 

        return torch.LongTensor(torch_data_x), torch.LongTensor(sample_data_y)

         

if __name__ == "__main__":
    print(vocab_check["wabalabadubdub"])
    print(len(vocab_check))
    data = NER_ConLL(mode="train", vocab=vocab_check, maxlen=128)
    for x, y in data:
        print(x.shape, y.shape)
        break
