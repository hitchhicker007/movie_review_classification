import tensorflow as tf 
from tensorflow import keras
import numpy as np 

data = keras.datasets.imdb

word_index = data.get_word_index()

word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value,key) for (key, value) in word_index.items()])

def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])

def review_encode(s):
	encoded = [1]

	for word in s:
		if word.lower() in word_index:
			encoded.append(word_index[word.lower()])
		else:
			encoded.append(2)

	return encoded

model = keras.models.load_model('trained_model')

text = "Man! I thought Infinity War was off the hook, but this one... A perfect compliment to a two part epic. It was what I thought it was going to be, a lot of talk, a lot of planning, but amazingly it was not boring. Not for one second! Infinity War was the moment we were all waiting for and then they tell us they are going to split it into two films. End Game could have gone south with this choice, but it did not. It's always exciting, especally if you are a comic book fan or have been watching these movies for now eleven years. Don't want to say too much, but shine a light on Robert Downey Jr. who got to perfect his Iron Man persona to the point where Hugh Jackman can't even surpass him. Captain American is giving all us FanBoys everything we ever wanted to see and Thor...underperforms with hilarious results (and hopefully will create 2019's most popular costume for a certain type of man). Loved Mark Rufflo in this film and Thanos baby, still the baddest villain around. It's fun it's dramatic, it's like Lord of the Rings with one less movie. Three hours well worth the ticket. Stan Lee would be proud! Nuff said!"

text = text.replace(",", "").replace(".", "").replace("(", "").replace(")", "").replace(":", "").replace("\"","").strip().split(" ")
encode = review_encode(text)
encode = keras.preprocessing.sequence.pad_sequences([encode], value=word_index["<PAD>"], padding="post", maxlen=250) # make the data 250 words long
predict = model.predict(encode)
print(text)
print(encode)
print(predict[0])