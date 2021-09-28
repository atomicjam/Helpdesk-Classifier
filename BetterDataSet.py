import json
import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import matplotlib.pyplot as plt
import io
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Variable setup
vocab_size = 10000
embedding_dim = 64
max_length = 100
num_epochs = 30
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"

sentences = []
labels = []
validation_sentences = []
validation_labels = []

# Load dataset
with open("training.json", 'r') as f:
    datastore = json.load(f)

# Setup sentences/labels from dataset
for item in datastore:
    sentences.append(item['title'])
    labels.append(item['helpdesk'])

# Load dataset
with open("validation.json", 'r') as f:
    datastore = json.load(f)

# Setup sentences/labels from dataset
for item in datastore:
    validation_sentences.append(item['title'])
    validation_labels.append(item['helpdesk'])

# Split dataset into training and testing sets
training_sentences = sentences
testing_sentences = validation_sentences
training_labels = labels
testing_labels = validation_labels

# Check Split dataset
print("\nTraining size:", len(training_sentences))
print("\nTesting size:", len(testing_sentences))

# Tokenize sentences and labels
tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)
word_index = tokenizer.word_index
print("\nTokenized Word Index Length:", len(word_index))
print("\nFirst 10 Tokens:",dict(list(word_index.items())[0:10]))
print("\nThe unique labels that we have in the training labels:", set(labels))

# Sequence and pad sentences
training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded_sequences = pad_sequences(training_sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)
testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded_sequences = pad_sequences(testing_sequences, padding=padding_type, truncating=trunc_type, maxlen=max_length)
# Need this block to get working with Tensorflow 2.x
training_padded_sequences = np.array(training_padded_sequences)
testing_padded_sequences = np.array(testing_padded_sequences)

# Output example 
print("\n10th Item Sequence:",training_sequences[10])
print("\n10th Item Length:",len(training_sequences[10]))
print("\n10th Item Padded Length:",len(training_padded_sequences[10]))
print("\nTraining Padded Shape:",training_padded_sequences.shape)
print("\nTesting Padded Shape:",testing_padded_sequences.shape)

#Tokenize Labels
training_label_seq = np.array(label_tokenizer.texts_to_sequences(training_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(testing_labels))

# Define Model 
model = tf.keras.Sequential([
        tf.keras.layers.Embedding(vocab_size, embedding_dim),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
        #tf.keras.layers.Conv1D(128,5, activation='relu'),
        #tf.keras.layers.GlobalMaxPooling1D(),
        tf.keras.layers.Dense(embedding_dim, activation='relu'),
        tf.keras.layers.Dense(12, activation='softmax')
])
model.summary()
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train model 
history = model.fit(training_padded_sequences, training_label_seq, epochs=num_epochs, validation_data=(testing_padded_sequences, validation_label_seq),  verbose=2)

# Plot Training
def plot_graphs(history, string):
  plt.plot(history.history[string])
  plt.plot(history.history['val_'+string])
  plt.xlabel("Epochs")
  plt.ylabel(string)
  plt.legend([string, 'val_'+string])
  plt.show()
plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# weights and words for vector
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#e = model.layers[0]
#weights = e.get_weights()[0]

# Save vector files for plotting on https://projector.tensorflow.org/
#out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
#out_m = io.open('meta.tsv', 'w', encoding='utf-8')
#for word_num in range(1, vocab_size):
#  word = reverse_word_index[word_num]
#  embeddings = weights[word_num]
#  out_m.write(word + "\n")
#  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
#out_v.close()
#out_m.close()

#Predict 
sentence = ["my room needs painting", "Password and MFA", "my toilet is broken", "chairs need moving", "Can i have a new toner for my printer", "how do i set the voicemail on my phone", "we have a flood in room 45", "parking needed", "bush in garden overgrown", "projector in room is dim", "room setup for meeting"]
helpdesks = ["empty","IT & AV Support","Websites","Audio Visual","Maintance","Estates Projects","Grounds & Gardens","Porters","Photographer","Data Manager","Theatre Techs","Medical Centre","Archives","Security"]
sequences = tokenizer.texts_to_sequences(sentence)
padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
predict = model.predict(padded)
for count, prediction in enumerate(predict):
    print("\n", sentence[count], "\n", int(labels[np.argmax(prediction)]), helpdesks[int(labels[np.argmax(prediction)])])
    
# Save Model for TF.JS
tfjs.converters.save_keras_model(model, 'tfjs')
model.save('model')
