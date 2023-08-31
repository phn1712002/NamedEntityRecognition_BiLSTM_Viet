import argparse, warnings, wandb, tensorflow
from Tools.Json import loadJson
from Tools.Callbacks import CreateCallbacks
from Tools.Weights import loadNearest, loadWeights
from Tools.TFLite import convertModelKerasToTflite
from Dataset.Createdataset import DatasetNERBiLSTM
from Architecture.Pipeline import PipelineNERBiLSTM 
from Optimizers.OptimizersNERBiLSTM import CustomOptimizers
from Architecture.Model import NERBiLSTM

# Environment Variables
PATH_CONFIG = './config.json'
PATH_DATASET = './Dataset/'
PATH_LOGS = './Checkpoint/logs/'
PATH_TENSORBOARD = './Checkpoint/tensorboard/'
PATH_TFLITE = './Checkpoint/export/'
ENCODEING = 'UTF-8'

# Argparse
parser = argparse.ArgumentParser()
parser.add_argument('--pretrain_config', type=bool, default=False, help='Pretrain model BiLSTM in logs training in dataset')
parser.add_argument('--path_file_pretrain', type=str, default='', help='Path file pretrain model')
parser.add_argument('--export_tflite', type=bool, default=False, help='Export to tflite')
args = parser.parse_args()

# Get config
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_wandb', 'config_model', 'config_opt', 'config_other', 'config_train']
    if all(key in config for key in keys_to_check):
        config_wandb = config['config_wandb']
        config_model = config['config_model']
        config_opt = config['config_opt']
        config_other = config['config_other']
        config_train = config['config_train']
    else:
        raise RuntimeError('Error config')
        
# Turn off warning
if not config_other['warning']:
    warnings.filterwarnings('ignore')
    
# Load dataset
train_raw_dataset, dev_raw_dataset, test_raw_dataset = DatasetNERBiLSTM(path=PATH_DATASET, encoding=config_model['decode'])()

# Init vocab and tags
vocab_map = tensorflow.keras.preprocessing.text.Tokenizer(lower=True, split=' ', filters=' ', oov_token='UNK')
tags_map = tensorflow.keras.preprocessing.text.Tokenizer(lower=True, split=' ', filters=' ', oov_token='UNK')
vocab_map.fit_on_texts(train_raw_dataset[0])
tags_map.fit_on_texts(train_raw_dataset[1])

# Create pipeline 
pipeline = PipelineNERBiLSTM(vocab_map=vocab_map, 
                                    tags_map = tags_map, 
                                    config_model=config_model)

train_dataset = PipelineNERBiLSTM(vocab_map=vocab_map, 
                                    tags_map = tags_map, 
                                    config_model=config_model)(dataset=train_raw_dataset, batch_size=config_train['batch_size_train'])

dev_dataset = PipelineNERBiLSTM(vocab_map=vocab_map, 
                                  tags_map=tags_map,
                                  config_model=config_model)(dataset=dev_raw_dataset, batch_size=config_train['batch_size_dev'])



# Create optimizers
opt_biLSTM = CustomOptimizers(**config_opt)()

# Callbacks
callbacks_NER = CreateCallbacks(PATH_TENSORBOARD=PATH_TENSORBOARD, 
                                PATH_LOGS=PATH_LOGS, 
                                config=config, 
                                train_dataset=train_dataset, 
                                dev_dataset=dev_dataset, 
                                pipeline=pipeline)

# Create model
ner = NERBiLSTM(vocab_map=vocab_map, 
                 tags_map=tags_map,
                 opt=opt_biLSTM,
                 **config_model).build(summary=config_other['summary'])

# Pretrain
if args.pretrain_config:
    if args.path_file_pretrain == '':
        ner = loadNearest(class_model=ner, path_folder_logs=PATH_LOGS)
    else: 
        ner = loadWeights(class_model=ner, path=args.path_file_pretrain)

# Train model
ner.fit(train_dataset=train_dataset, 
        dev_dataset=dev_dataset, 
        callbacks=callbacks_NER,
        epochs=config_train['epochs'])

# Export to tflite
if args.export_tflite:
    convertModelKerasToTflite(class_model=ner, path=PATH_TFLITE)
    
# Off Wandb
wandb.finish()