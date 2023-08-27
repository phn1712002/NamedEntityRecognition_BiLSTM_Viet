import pandas as pd
from Tools.Json import saveJson, loadJson
from Tools.NLP import MapToIndex


path = './Dataset/'
vocab = []
with open(path + 'raw/input_string.txt', 'r', encoding='utf-8') as file:
    for line in file:
        vocab += line.strip().split()
        
tags = []
with open(path + 'raw/file_name_lable.txt', 'r', encoding='utf-8') as file:
    for line in file:
        tags += line.strip().lower().split()

word_mti = MapToIndex().settingWithList(vocab)
tag_mti = MapToIndex().settingWithList(tags)


saveJson(path=path + "raw/config_vocab.json", data=word_mti.getMap())
saveJson(path=path + "raw/config_tags.json", data=tag_mti.getMap())
