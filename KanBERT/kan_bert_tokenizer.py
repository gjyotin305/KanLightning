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
        self.vocab_size = vocab_size
        if vocab_dict:
            self.vocab_dict = vocab_dict
        else:
            self.vocab_dict = initial_dict

        self.vocab_to_id = {w:i for i, w in self.vocab_dict.items()}

    def encode(self, text: str) -> List[int]:
        split_word = nltk.word_tokenize(text)
        final_encode = []
        
        for x in split_word:
            final_encode.append(self.vocab_dict[x])
        
        return final_encode
    
    def encode_plus(self, 
                    text: str, 
                    return_tensor=True) -> Dict[List[Tensor], List[Tensor]]:
        split_word = nltk.word_tokenize(text)
        final_encode = [self.vocab_dict["[CLS]"]]
        
        for x in tqdm(split_word):
            final_encode.append(self.vocab_dict[x])
        
        final_encode.append(self.vocab_dict["[SEP]"])
        
        print(len(final_encode))

        if len(final_encode) > self.max_length:
            print("Truncation has to take place")
            final_encode = final_encode[:self.max_length]
            final_encode.append(self.vocab_dict["[SEP]"])

        if len(final_encode) < self.max_length:
            print("Padding has to take place.")
            pad_extension = [self.vocab_dict["[PAD]"]]*(self.max_length-len(final_encode))
            final_encode.extend(pad_extension)


        segment_ids = [0]*len(final_encode)

        return {
            "input_ids": torch.LongTensor(final_encode),
            "segment_ids": torch.Tensor(segment_ids)
        } 


    def decode(self, text_list: List[int]) -> List[str]:
        decode_string = []

        for x in tqdm(text_list):
            decode_string.append(self.vocab_to_id[x])
        
        return decode_string 

    def ingest_vocab_batch(self, text: List[str]) -> None:
        check_length = len(self.vocab_dict)

        if check_length > len(initial_dict):
            for i, x in enumerate(tqdm(text)):
                split_word = nltk.word_tokenize(x)
                for y in split_word:
                    self.vocab_dict[y] = i + check_length

        elif check_length == len(initial_dict):

            for i, x in enumerate(tqdm(text)):
                split_word = nltk.word_tokenize(x)
                for y in split_word:
                    self.vocab_dict[y] = i + len(initial_dict)

        self.vocab_to_id = {w:i for i, w in self.vocab_dict.items()}

        print(f"The length of the vocabulary is {len(self.vocab_dict)}")


    
        

