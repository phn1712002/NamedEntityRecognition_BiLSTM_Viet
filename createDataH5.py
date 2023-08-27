
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

"""
import pandas as pd
from sklearn.model_selection import train_test_split
from Tools.Json import saveJson
from Tools.H5 import saveH5
from Tools.NLP import MapToIndex


path = './Dataset/'
df = pd.read_csv(path + 'raw/ner_dataset.csv', encoding = "ISO-8859-1")
df = df.fillna(method = 'ffill')
df['Word'] = df['Word'].apply(lambda x: x.lower())
df = df.drop(['POS'], axis=1)


agg = lambda s: [(w, p) for w, p in zip(s['Word'].values.tolist(), s['Tag'].values.tolist())]
grouped = df.groupby("Sentence #").apply(agg)
sentences = [s for s in grouped]

word_mti = MapToIndex().settingWithList(list(df['Word']))
tag_mti = MapToIndex().settingWithList(list(df['Tag']))

dataX = []
dataY = []
for s in sentences:
    tempX = ""
    tempY = ""
    for X, Y in s:
        tempX += X + " "
        tempY += Y + " "
    dataX.append(str(tempX[:-1]))
    dataY.append(str(tempY[:-1]))

saveJson(path=path + "raw/config_vocab.json", data=word_mti.getMap())
saveJson(path=path + "raw/config_tags.json", data=tag_mti.getMap())
X_train, X_dev, y_train, y_dev = train_test_split(dataX, dataY, test_size=0.3)


train_dataset = {
    'X': X_train,
    'Y': y_train
}

dev_dataset = {
    'X': X_dev,
    'Y': y_dev
}

saveH5(path=path + 'raw/train_dataset.h5', data=train_dataset)
saveH5(path=path + 'raw/dev_dataset.h5', data=dev_dataset)
"""