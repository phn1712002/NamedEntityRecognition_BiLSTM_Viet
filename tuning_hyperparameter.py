# Environment variables
PATH_CONFIG = './tuning_hyperparameter.json'
PATH_DATASET = './Dataset/'

# Get config
from Tools.Json import loadJson
config = loadJson(PATH_CONFIG)
if not config == None:
    keys_to_check = ['config_wandb', 'config_sweep', 'config_dataset', 'config_other']
    if all(key in config for key in keys_to_check):
        config_wandb = config['config_wandb']
        config_sweep = config['config_sweep']
        config_dataset = config['config_dataset']
        config_other = config['config_other']
    else:
        raise RuntimeError('Error config')

# Create Sweep WandB
import wandb, os
os.environ['WANDB_API_KEY'] = config_wandb['api_key']
wandb.login()
sweep_id = wandb.sweep(config_sweep, project=config_wandb['project'])
        
# Turn off warning
import warnings
if not config_other['warning']:
    warnings.filterwarnings('ignore')
    
# Load dataset
from Dataset.Createdataset import DatasetNERBiLSTM
train_raw_dataset, dev_raw_dataset, test_raw_dataset = DatasetNERBiLSTM(path=PATH_DATASET)()

# Split dataset
from Tools.TuningHyper import splitDataset
train_raw_dataset = splitDataset(train_raw_dataset, size_dataset=config_dataset['size_dataset'])
dev_raw_dataset = splitDataset(dev_raw_dataset, size_dataset=config_dataset['size_dataset'])

# Load config
config_vocab = loadJson(config_dataset['path_json_vocab'])
config_tags = loadJson(config_dataset['path_json_tags'])

# Init vocab and tags
from Tools.NLP import MapToIndex
vocab_map = MapToIndex().settingWithDict(config_vocab)
tags_map = MapToIndex().settingWithDict(config_tags)


# Tuning Hyperparameter
from Architecture.Model import NERBiLSTM
from Architecture.Pipeline import PipelineNERBiLSTM 
from Optimizers.OptimizersNERBiLSTM import CustomOptimizers
from wandb.keras import WandbCallback
def tuningHyperparamtrer(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Create all config
        config_model = {
            "name": "NERBiLSTM_Tuning",
            "max_len": config.max_len,
            "embedding_dim": config.embedding_dim,
            "num_layers": config.num_layers,
            "hidden_size": config.hidden_size,
            "rate_dropout": config.rate_dropout
            }
        
        config_opt = {
            "learning_rate": pow(10, config.learning_rate)
        }
        
        config_dataset = {
            "batch_size_train": config.batch_size_train,
            "batch_size_dev": config.batch_size_dev
        }
        
        config_train = {
            "epochs": config.epochs
        }
        
        # Create pipeline 
        train_dataset = PipelineNERBiLSTM(vocab_map=vocab_map, 
                                            tags_map = tags_map,
                                            dataset=train_raw_dataset, 
                                            batch_size=config_dataset['batch_size_train'],
                                            config_model=config_model)()

        dev_dataset = PipelineNERBiLSTM(vocab_map=vocab_map, 
                                        tags_map=tags_map,
                                        dataset=dev_raw_dataset, 
                                        batch_size=config_dataset['batch_size_dev'],
                                        config_model=config_model)()

        # Create optimizers
        opt_biLSTM = CustomOptimizers(**config_opt)()

        # Create model
        ner = NERBiLSTM(vocab_map=vocab_map, 
                        tags_map=tags_map,
                        opt=opt_biLSTM,
                        **config_model).build()

        # Train model
        ner.fit(train_dataset=train_dataset, 
                dev_dataset=dev_dataset,
                epochs=config_train['epochs'],
                callbacks=[WandbCallback(log_weights=True, 
                                         log_gradients=True, 
                                         save_model=False, 
                                         training_data=train_dataset,
                                         validation_data=dev_dataset,
                                         log_evaluation=True)])

# Tuning 
wandb.agent(sweep_id, tuningHyperparamtrer, count=config_wandb['count'])