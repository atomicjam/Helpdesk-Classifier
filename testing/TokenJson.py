import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

with open("query_result.json", 'r') as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['title'])
    labels.append(item['label'])


tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\nWord Index:", word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print("\nWord Sequences:", sequences[0])

padded_sequences = pad_sequences(sequences, padding='post', truncating='post', maxlen=255)
print("\nPadded Sequences:", padded_sequences[0])
print("\nPadded Shape:", padded_sequences.shape)






