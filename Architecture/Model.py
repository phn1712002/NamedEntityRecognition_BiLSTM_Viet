import tensorflow as tf
import numpy as np
from keras.utils import to_categorical, pad_sequences
from keras import optimizers, losses, Model, Input
from keras.preprocessing.text import Tokenizer
from keras.layers import LSTM, Embedding, Dense, Bidirectional, TimeDistributed, Dropout

class CustomModel():
    def __init__(self, vocab_map:Tokenizer, tags_map:Tokenizer, model=Model(), loss=losses.CategoricalCrossentropy(from_logits=True), opt=optimizers.Adam()):
        self.vocab_map = vocab_map
        self.tags_map = tags_map
        self.model = model
        self.loss = loss
        self.opt = opt
        
    def build(self, summary=False):
        pass
    
    def getConfig(self):
        pass
    
    def fit(self, train_dataset, dev_dataset=None, epochs=1, callbacks=None):
        pass
    
    def predict(self, input):
        pass
    
class NERBiLSTM(CustomModel):
    def __init__(self, 
                 vocab_map:Tokenizer,
                 tags_map:Tokenizer,
                 name="NERBiLSTM", 
                 max_len=50,
                 embedding_dim=20,
                 num_layers=1,
                 hidden_size=512,
                 rate_dropout=0.5,
                 decode='utf-8',
                 opt=optimizers.Adam(),
                 loss=losses.CategoricalCrossentropy(from_logits=True)):
        super().__init__(vocab_map=vocab_map, tags_map=tags_map, model=None, opt=opt, loss=loss)
        self.name = name
        self.max_len = max_len
        self.vocab_size = len(vocab_map.index_word) + 1
        self.embedding_dim = embedding_dim
        self.num_tags = len(tags_map.index_word) + 1
        self.num_layers = num_layers
        self.hidden_size= hidden_size
        self.rate_dropout = rate_dropout
        self.decode = decode

    def build(self, summary=False):
        strategy = tf.distribute.MirroredStrategy()
        with strategy.scope():
            input = Input(shape=(self.max_len, ), name="input")
            
            X = Embedding(input_dim=self.vocab_size, output_dim = self.embedding_dim, input_length = self.max_len, mask_zero=False, name="embdding")(input)
            X = Dropout(self.rate_dropout)(X)
            for i in range(1, self.num_layers + 1):
                X = Bidirectional(LSTM(units=self.hidden_size, return_sequences=True, recurrent_dropout=self.rate_dropout), name=f"Bidirectional_{i}")(X)
                
            X = LSTM(units=self.hidden_size * 2, return_sequences=True, recurrent_dropout=self.rate_dropout, name="LSTM")(X) 
            output = TimeDistributed(Dense(self.num_tags, activation = 'softmax'), name="TimeDistributed")(X)
        
            model = Model(inputs=input, outputs=output, name=self.name)
         
            model.compile(optimizer=self.opt, loss=self.loss, metrics=["accuracy"])
        if summary:
            model.summary()
            
        self.model = model
        return self
    
    def fit(self, train_dataset, dev_dataset=None, epochs=1, callbacks=None):
        
        self.model.fit(x=train_dataset,
                       validation_data=dev_dataset,
                       epochs=epochs,
                       callbacks=callbacks)
        return self
        
    def predict(self, input):
        
        input_tf, input_size = self.formatInput(input=input)
        output_tf  = self.model.predict_on_batch(input_tf)
        output = self.formatOutput(output_tf=output_tf, input_size=input_size)
        return  list(zip(input.split(), output))
    
    def formatInput(self, input):
        input_size = len(input.split())
        input_tf = self.encoderSeq(tf.convert_to_tensor(input))
        input_tf = tf.expand_dims(input_tf, axis=0)
        return input_tf, input_size
    
    def formatOutput(self, output_tf, input_size):
        
        output = self.decoderLable(output_tf)
        output = str(output[0]).split()
        output = output[:input_size]
        return output
    
    def getConfig(self):
        return {
            "name": self.name,
            "max_len": self.max_len,
            "embedding_dim": self.embedding_dim,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "rate_dropout": self.rate_dropout,
            "decode": self.decode
        }
        
    def encoderSeq(self, seq=None):
        
        seq = seq.numpy().decode(self.decode)
        seq = self.vocab_map.texts_to_sequences([seq])
        seq = tf.convert_to_tensor(seq)
        seq = pad_sequences(seq, value=0, maxlen=self.max_len, padding='post')
        seq = tf.squeeze(seq)
        return tf.cast(seq, dtype=tf.int32)
    
    def encoderLable(self, lable=None):
        
        lable = lable.numpy().decode(self.decode)
        lable = self.tags_map.texts_to_sequences([lable])
        lable = tf.convert_to_tensor(lable)
        lable = pad_sequences(lable, value=0, maxlen=self.max_len, padding='post')
        lable = tf.squeeze(lable)
        lable = [to_categorical(i, num_classes=self.num_tags, dtype='int32') for i in lable.numpy()]
        return tf.convert_to_tensor(lable, dtype=tf.int32)
    
    def decoderLable(self, output_tf=None):
        
        output_tf = tf.math.argmax(output_tf, axis=-1)
        output = tf.squeeze(output_tf).numpy()
        output = self.tags_map.sequences_to_texts([output])
        return output
    
    def decoderSeq(self, input_tf=None):
        
        input = tf.squeeze(input_tf).numpy()
        input = self.vocab_map.sequences_to_texts([input])
        return input
    
class NERBiLSTM_tflie(NERBiLSTM):
    def __init__(self, vocab_map, tags_map, config_model=None):
        super().__init__(vocab_map=vocab_map, tags_map=tags_map, opt=None, loss=None, **config_model)
        
    def predict(self, input):
        input_tf, input_size = super().formatInput(input)
        output_tf  = self.invoke(input_tf)
        output = super().formatOutput(output_tf=output_tf, input_size=input_size)
        return list(zip(input.split(), output))
    
    def invoke(self, input_tf):
        pass