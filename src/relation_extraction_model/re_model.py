import os
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
import transformers
from transformers import TFAutoModel, AutoTokenizer
from tqdm.notebook import tqdm
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, processors
from typing import List

class RelationExtractModel(object):
    def __init__(self, transformer_dir:str, pickle_path:str):
        self.re_model = self.load_model(pickle_path, transformer_dir, 512)
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/biomed_roberta_base")

    @staticmethod
    def build_model(transformer:TFAutoModel, max_len:int=256):
        """
        Constructs a BERT model. See link below for more
        https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras
        """
        input_ids = Input(shape=(max_len, ), dtype=tf.int32)

        x = transformer(input_ids)[0]
        x = x[:, 0, :]
        x = Dense(1, activation='sigmoid', name='sigmoid')(x)

        # BUILD AND COMPILE MODEL
        model = Model(inputs=input_ids, outputs=x)
        model.compile(
            loss='binary_crossentropy', 
            metrics=['accuracy'], 
            optimizer=Adam(lr=1e-5)
        )
        return model

    def save_model(self, model, pickle_dir:str, transformer_dir:str):
        """
        Special function to save a keras model that uses a transformer layer
        """
        transformer = self.re_model.layers[1]
        transformer.save_pretrained(transformer_dir)
        sigmoid = self.re_model.get_layer('sigmoid').get_weights()
        pickle.dump(sigmoid, open(pickle_dir, 'wb'))

    def load_model(self, pickle_path:str, transformer_dir:str='transformer', max_len=512):
        """
        Special function to load a keras model that uses a transformer layer
        """
        transformer = TFAutoModel.from_pretrained(transformer_dir)
        model = self.build_model(transformer, max_len=max_len)
        sigmoid = pickle.load(open(pickle_path, 'rb'))
        model.get_layer('sigmoid').set_weights(sigmoid)
        
        return model

    def regular_encode(self, texts:List[str], maxlen=512)->np.array:
        """"""
        enc_di = self.tokenizer.batch_encode_plus(
            texts, 
            return_attention_masks=False, 
            return_token_type_ids=False,
            pad_to_max_length=True,
            max_length=maxlen
        )
        return np.array(enc_di['input_ids'])
    
    def predict(self, texts:List[str], threshold=.4)->List[bool]:
        """
        Function that takes a list of strings and returns 
        a boolean value for each that string contains a drug treatment pair.
        """
        if type(texts) == str:
            new_list = [] 
            new_list.append(texts)
            texts = new_list
        output = self.regular_encode(texts)
        model_output = self.re_model.predict(output)
        return list(map(lambda x: x>threshold, model_output))