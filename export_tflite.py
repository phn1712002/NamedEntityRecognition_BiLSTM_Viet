import argparse, tensorflow
from Tools.Json import loadJson
from Tools.Callbacks import CreateCallbacks
from Tools.Weights import loadNearest, loadWeights
from Tools.TFLite import convertModelKerasToTflite
from Dataset.Createdataset import DatasetNERBiLSTM
from Architecture.Model import NERBiLSTM

# Environment Variables
PATH_CONFIG = './config.json'
PATH_DATASET = './Dataset/'
PATH_LOGS = './Checkpoint/logs/'
PATH_TFLITE = './Checkpoint/export/'

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--path_file_pretrain', type=str, default='', help='Path file pretrain model')
args = parser.parse_args()

# Get config
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_model', 'config_other']
    if all(key in config for key in keys_to_check):
        config_model = config['config_model']
        config_other = config['config_other']
    else:
        raise RuntimeError('Error config')
        
# Load dataset
train_raw_dataset, dev_raw_dataset, test_raw_dataset = DatasetNERBiLSTM(path=PATH_DATASET, encoding=config_model['decode'])()

# Init vocab and tags
vocab_map = tensorflow.keras.preprocessing.text.Tokenizer(lower=True, split=' ', filters=' ', oov_token='UNK')
tags_map = tensorflow.keras.preprocessing.text.Tokenizer(lower=False, split=' ', filters=' ', oov_token='UNK')
vocab_map.fit_on_texts(train_raw_dataset[0])
tags_map.fit_on_texts(train_raw_dataset[1])

# Create model
ner = NERBiLSTM(vocab_map=vocab_map, 
                tags_map=tags_map,
                **config_model).build(summary=config_other['summary'])
# Load Weights
if args.path_file_pretrain == '':
    ner = loadNearest(class_model=ner, path_folder_logs=PATH_LOGS)
else: 
    ner = loadWeights(class_model=ner, path=args.path_file_pretrain)

# Export tflite
convertModelKerasToTflite(class_model=ner, path=PATH_TFLITE)
        