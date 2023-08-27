
from typing import Any


class importTXT():
    def __init__(self, path):
         self.path = path

    def __call__(self, *args: Any, **kwds: Any):
        vocab = []
        with open(self.path + 'input_string.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    vocab.append(line.strip().lower())
                    
        tags = []
        with open(self.path + 'file_name_lable.txt', 'r', encoding='utf-8') as file:
                for line in file:
                    tags.append(line.strip().lower())
                
                    
        return vocab, tags