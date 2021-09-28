import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

sentences = [
    'I love my dog',
    'I love my cat',
    'You love my Dog!',
    'Do you think my dog is amazing?'
]

tokenizer = Tokenizer(num_words = 100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print("\nWord Index:", word_index)

sequences = tokenizer.texts_to_sequences(sentences)
print("\nWord Sequences:", sequences)

padded_sequences = pad_sequences(sequences, padding='post', truncating='post', maxlen=255)
print("\nPadded Sequences:", padded_sequences)

test_data = [
    'i really love my dog',
    'my dog loves my manatee',
    'do you love my dog?',
    'i think my dog loves my cat'
]

test_sequence = tokenizer.texts_to_sequences(test_data)
print("\nTest Sequences:", test_sequence)

padded_test_sequence = pad_sequences(test_sequence, padding='post', truncating='post', maxlen=255)
print("\nPadded Test Sequences:", padded_test_sequence)

