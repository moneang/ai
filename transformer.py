import pandas as pd
import numpy as np
import os
import re
from pprint import pprint
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.preprocessing.sequence import pad_sequences

data=pd.read_csv('ChatbotData .csv')
q_data=list(data['Q'])
a_data=list(data['A'])
questions=[]
answers=[]
# pprint(q_data)
for i,j in zip(q_data,a_data):
    txt=re.sub(r'([?.!,])',r' \1',i)
    questions.append(txt)
    txt=re.sub(r'([!?,.])',r' \1',j)
    answers.append(txt)
# print(answers)
tokenizer=tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
    questions+answers,target_vocab_size=2**13)
start_token,end_token=[tokenizer.vocab_size], [tokenizer.vocab_size+1]
vocab_size=tokenizer.vocab_size+2
print(start_token)
print(end_token)
print(vocab_size)
inputs=[]
outputs=[]
for i,j in zip(questions,answers):
    txt=start_token+tokenizer.encode(i)+end_token
    inputs.append(txt)
    txt=start_token+tokenizer.encode(j)+end_token
    outputs.append(txt)
inputs=pad_sequences(inputs, maxlen=40, padding= 'post')
outputs=pad_sequences(outputs,maxlen=40, padding='post')
print(inputs.shape)
print(inputs[1])
