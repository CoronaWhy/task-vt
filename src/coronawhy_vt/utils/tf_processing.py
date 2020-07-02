# -*- coding: utf-8 -*-

"""Temporary storage for utils used in the processing of local data structures using tensorflow."""

import os
import pickle

import numpy as np
import tensorflow as tf
# import transformers
# from kaggle_datasets import KaggleDatasets
# from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
# from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors
# from tqdm.notebook import tqdm
from transformers import TFAutoModel
# from transformers import AutoTokenizer, TFAutoModel


def touch_dir(dirname):
    """Temporary placeholder."""
    os.makedirs(dirname, exist_ok=True)


def build_model(transformer, max_len=256):
    """From https://www.kaggle.com/xhlulu/jigsaw-tpu-distilbert-with-huggingface-and-keras."""
    input_ids = Input(shape=(max_len, ), dtype=tf.int32)

    x = transformer(input_ids)[0]
    x = x[:, 0, :]
    x = Dense(1, activation='sigmoid', name='sigmoid')(x)

    # BUILD AND COMPILE MODEL
    model = Model(inputs=input_ids, outputs=x)
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=1e-5))
    return model


def save_model(model, transformer_dir='transformer'):
    """Load a keras model containing a transformer layer."""
    transformer = model.layers[1]
    touch_dir(transformer_dir)
    transformer.save_pretrained(transformer_dir)
    sigmoid = model.get_layer('sigmoid').get_weights()
    pickle.dump(sigmoid, open('sigmoid.pickle', 'wb'))


def load_model(pickle_path, transformer_dir='transformer', max_len=512):
    """Load a keras model containing a transformer layer."""
    transformer = TFAutoModel.from_pretrained(transformer_dir)
    model = build_model(transformer, max_len=max_len)
    sigmoid = pickle.load(open(pickle_path, 'rb'))
    model.get_layer('sigmoid').set_weights(sigmoid)

    return model


def regular_encode(texts, tokenizer, maxlen=512):
    """Encode texts with provided tokenizer."""
    enc_di = tokenizer.batch_encode_plus(
        texts,
        return_attention_masks=False,
        return_token_type_ids=False,
        pad_to_max_length=True,
        max_length=maxlen)

    return np.array(enc_di['input_ids'])


def magic_classifier_function(model, sentence, tokenizer):
    """Run binary classifier on encoded sentece."""
    encode_text = regular_encode(sentence, tokenizer)
    preds = model.predict(encode_text)
    boolean_decision = (preds >= .4)
    return boolean_decision
