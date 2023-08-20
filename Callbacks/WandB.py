import wandb
import numpy as np 
import tensorflow as tf
from keras.callbacks import Callback
from Tools.Weights import getPathWeightsNearest
from Architecture.Pipeline import PipelineNERBiLSTM

class CustomCallbacksWandB(Callback):
    def __init__(self, pipeline: PipelineNERBiLSTM, path_logs='./Checkpoint/logs/', dev_dataset=None):
        super().__init__()
        self.dev_dataset = dev_dataset
        self.path_logs = path_logs
        self.pipeline = pipeline
        self.__last_name_update = None
        
    def on_epoch_end(self, epoch: int, logs=None):
        
        # Print 1 mẫu 
        tableOutputPredict = wandb.Table(columns=["Epoch", "Input", "Output"])
        for X, _ in self.dev_dataset.take(1):
            if not X.shape[0] == 1:
                index = np.random.randint(low=0, high=X.shape[0] - 1)
                X = X[index]
                X = tf.expand_dims(X, axis=0)
                
        Y = self.pipeline.predictInCallbacks(self.model, X)  
        X = self.pipeline.decoderSeq(X)
        
        tableOutputPredict.add_data(epoch + 1, X, Y)
        wandb.log({'Predict': tableOutputPredict})
        
        # Cập nhật file weights model to cloud wandb
        path_file_update = getPathWeightsNearest(self.path_logs)
        if self.__last_name_update != path_file_update: 
            self.__last_name_update = path_file_update
            wandb.save(path_file_update)
        
        