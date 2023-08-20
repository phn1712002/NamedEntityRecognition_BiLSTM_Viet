import tensorflow as tf
from Tools.Json import saveJson
from Architecture.Model import CustomModel

def convertModelKerasToTflite(class_model: CustomModel, path="./Checkpoint/export/"):
    
    # Convert to tflite
    converter = tf.lite.TFLiteConverter.from_keras_model(class_model.model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.experimental_new_converter=True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    
    # Get config model
    config_model = class_model.getConfig()
    config_vocab = class_model.vocab_map.getMap()
    config_tags = class_model.tags_map.getMap()
    
    path_tflite = path + class_model.name + '.tflite' 
    path_json = path + class_model.name + '.json'
    
    saveJson(path=path_json, data={
        "config_model": config_model,
        "config_vocab": config_vocab,
        "config_tags": config_tags
    })
    tf.io.write_file(filename=path_tflite, contents=tflite_model)
    print(f"Export model to tflite filename:{path_tflite} and json:{path_json}")
    return True