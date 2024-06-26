import torch
from typing import (Dict, 
                    List)
from tqdm import tqdm
import nltk
from torch import Tensor
from .constants import initial_dict
import torch.nn as nn

class KanTokenizer:
    def __init__(self, 
                vocab_size: int = 4,
                max_length: int = 256,
                vocab_dict: dict =None ) -> None:
    
        super().__init__()
        self.max_length = max_length
        self.vocab_size = 5
        self.vocab_dict = initial_dict
        self.vocab_count = {i: 1 for i, w in initial_dict.items()}
        self.id_to_vocab = {w:i for i,w in initial_dict.items()}

    def encode(self, text: str) -> List[int]:
        split_word = nltk.word_tokenize(text)
        split_word = [w.lower() for w in split_word]
        final_encode = []
        
        for x in split_word:
            final_encode.append(self.vocab_dict[x])
        
        return final_encode
    
    def encode_plus(self, 
                    text: str, 
                    return_tensor=True) -> Dict[List[Tensor], List[Tensor]]:
        split_word = nltk.word_tokenize(text)
        split_word = [w.lower() for w in split_word]
        final_encode = [self.vocab_dict["[CLS]"]]
        
        for x in split_word:
            if x not in self.vocab_dict:
                final_encode.append(self.vocab_dict["[UNK]"])
            else:
                final_encode.append(self.vocab_dict[x])
        
        print(f"Initial encoding: {len(final_encode)}")

        if len(final_encode) == self.max_length:
            final_encode.append(self.vocab_dict["[SEP]"])

        if len(final_encode) > self.max_length:
            final_encode = final_encode[:self.max_length]
            final_encode.append(self.vocab_dict["[SEP]"])

        if len(final_encode) < self.max_length:
            final_encode.append(self.vocab_dict["[SEP]"])
            pad_extension = [self.vocab_dict["[PAD]"]]*(self.max_length-len(final_encode) + 1)
            final_encode.extend(pad_extension)

        print(f"Final Length is {len(final_encode)}")
        assert(len(final_encode) == self.max_length + 1)
        segment_ids = [0]*len(final_encode)

        return {
            "input_ids": torch.LongTensor(final_encode),
            "segment_ids": torch.LongTensor(segment_ids)
        } 


    def decode(self, text_list: List[int]) -> List[str]:
        decode_string = []

        for x in tqdm(text_list):
            decode_string.append(self.vocab_to_id[x])
        
        return decode_string 

    def ingest_vocab_batch(self, text: List[str]) -> None:
        for x in tqdm(text):
            split_word = nltk.word_tokenize(x)
            split_word = [w.lower() for w in split_word]
            for y in split_word:
                if y not in self.vocab_dict:
                    self.vocab_dict[y] = self.vocab_size
                    self.id_to_vocab[self.vocab_size] = y
                    self.vocab_count[y] = 1
                    self.vocab_size += 1
                else:
                    self.vocab_count[y] += 1
            
        print(f"Vocab size is {len(self.vocab_dict)}")


    
        

