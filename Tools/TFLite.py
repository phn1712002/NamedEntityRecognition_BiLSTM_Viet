import os
import tensorflow as tf
from Tools.Json import saveJson
from Architecture.Model import CustomModel

def convertModelKerasToTflite(class_model: CustomModel, path="./Checkpoint/export/"):
    
    if os.path.exists(path):
    
        # Convert to tflite
        converter = tf.lite.TFLiteConverter.from_keras_model(class_model.model)
        #converter.optimizations = [tf.lite.Optimize.DEFAULT]
        #onverter.experimental_new_converter=True
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        tflite_model = converter.convert()
        
        # Get config model
        config_model = class_model.getConfig()
        config_vocab = class_model.vocab_map.to_json()
        config_tags = class_model.tags_map.to_json()
        
        # Path config
        path_tflite = path + class_model.name + '.tflite' 
        path_json_vocab = path + class_model.name + '_vocab.json'
        path_json_tag = path + class_model.name + '_tags.json'
        path_json_config = path + class_model.name + '.json'
        
        # Save
        saveJson(path=path_json_config, data=config_model)
        saveJson(path=path_json_vocab, data=config_vocab, encoding=None)  
        saveJson(path=path_json_tag, data=config_tags, encoding=None) 
        tf.io.write_file(filename=path_tflite, contents=tflite_model)
        
        print(f"Export model to tflite filename:{path_tflite} and json:{path_json_config}")
    else:
        raise RuntimeError('Error path')