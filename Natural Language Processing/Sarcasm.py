!wget --no-check-certificate \
    https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sarcasm.json \
    -O /tmp/sarcasm.json
  
import json

with open("/tmp/sarcasm.json", 'r') as f:
    datastore = json.load(f)
    
import tensorflow as tf
import numpy as np

sentences = [] 
labels = []
urls = []
for item in datastore:
    sentences.append(item['headline'])
    labels.append(item['is_sarcastic'])
    urls.append(item['article_link'])
    
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)

word_index = tokenizer.word_index
print(len(word_index))
print(word_index)

sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences, padding='post')
print(padded[0])
print(padded.shape)

training_size = 20000

train_sentences = sentences[0:training_size]
train_labels = np.array(labels[0:training_size])
test_sentences = sentences[training_size:]
test_labels = np.array(labels[training_size:])

vocab_size = 10000
max_len=100

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<OOV>")
tokenizer.fit_on_texts(train_sentences)

word_index = tokenizer.word_index
seq_train = tokenizer.texts_to_sequences(train_sentences)
seq_pad_train = pad_sequences(seq_train,maxlen=max_len,truncating='post',padding='post')
seq_pad_train.shape

#test
seq_test = tokenizer.texts_to_sequences(test_sentences)
seq_pad_test = pad_sequences(seq_test,maxlen=max_len,truncating='post',padding='post')

from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Embedding, Dense, LSTM, Bidirectional, Dropout

model = Sequential([
    Embedding(vocab_size, 16),
    Bidirectional(LSTM(64,recurrent_dropout = 0.3 , dropout = 0.3, return_sequences = True)),
    Bidirectional(LSTM(32,recurrent_dropout = 0.1 , dropout = 0.1)),
    Dense(512, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics=['acc'])
history = model.fit(seq_pad_train,train_labels, epochs = 3, validation_data = (seq_pad_test, test_labels))

sentence = ["granny starting to fear spiders in the garedn might be real"]
sentence = tokenizer.texts_to_sequences(sentence)
sentence_pad = pad_sequences(sentence, maxlen=29, padding='post', truncating='post')

sentiment = model.predict(sentence_pad,batch_size=1,verbose = 2)[0]
print(sentiment)
if(np.argmax(sentiment) >= 0.5):
    print("Non-sarcastic")
elif (np.argmax(sentiment) < 0.5):
    print("Sarcasm")
    
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
