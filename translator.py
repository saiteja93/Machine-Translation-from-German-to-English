#Author: Saiteja Sirikonda

#Function: you enter a german sentence, returns the English translation of it. The length has to be kept less than 5, due to resource limitation thats what the model is trained on.


import re, string
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from nltk.translate.bleu_score import corpus_bleu
#import machine_trans.py as mt

def read_data(file_name):
    """
    #reads the data from the given file_name.
    """

    with open(file_name, "r+") as f:
        data = f.read()
    lines = data.strip().split("\n")
    pairs = [line.split("\t") for line in lines]
    #returns list of lists, with each entry 0 -> German Entry, 1 -> corresponding English
    #print (len(pairs))
    # for i in pairs:
    #     print (i)
    return pairs

def data_preprocessing(pairs):
    #removing punctuation, converting them to lowercase, removing the non-printable characters and removing the with numbers in them.
    cleaned = []
    for i in pairs:
        element = []
        for entry in i:
            #we start, by removing punctuation and turning to lowercase
            entry = re.sub(r'[^\w\s]',"",entry.lower())
            # we remove the entries with non-printable characters
            # print (entry)
            # input("removed punctuation")
            entry = "".join(filter(lambda x: x in string.printable, entry))
            # print (entry)
            # input("removed non printable characters")
            # removing numerical entries
            # temp = [word for word in entry if word.isalpha()]
            # entry = " ".join(temp)
            # print (entry)
            element.append(entry)
        cleaned.append(element)
    #print (len(cleaned))
    # for i in range(10):
    #     print (cleaned[i][0], cleaned [i][1])
    return np.array(cleaned)

def data_preprocessing_for_input(test):
    #we will be checking for punctuation, normalize it to lowercase and also remove unprintable characters incase entered.
    test = re.sub(r'[^\w\s]',"",test.lower())
    test = "".join(filter(lambda x: x in string.printable, test))
    return np.array(test)

def modify (train):
    for i in train[:,0]:
        i = " ".join(i.split()[::-1])
    return train

def tokenizer_object_creation(lines):
    #creates a tokenizer object for each language.
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def max_sentence_length(lines):
    #returns the length of the maximum sentence in Dataset.
    return max(len(line.split()) for line in lines)

def encode_input_sequences(tokenizer, length, lines):
	# integer encode sequences. We could be using Word2vec as well from Gensim. We make sure that all input sequences have the same length by padding the trailing positions with 0's.
	X = tokenizer.texts_to_sequences(lines)
	# pad sequences with 0 values
	X = pad_sequences(X, maxlen=length, padding='post')
	return X

def encode_to_onehot(sequences, vocab_size):
	ylist = list()
	for sequence in sequences:
		encoded = to_categorical(sequence, num_classes=vocab_size)
		ylist.append(encoded)
	y = np.array(ylist)
    #The data has to be reshaped because LSTM requires 3D data.
	y = y.reshape(sequences.shape[0], sequences.shape[1], vocab_size)
    #print ("the shape of y is {}".format(y.shape))
	return y

#the main aim is to come up with a Hindi-English translator, but let us start with this.
file_name = "deu.txt"
pairs = read_data(file_name)
processed_pairs = data_preprocessing(pairs)
processed_pairs = processed_pairs[:10000]

english_tokenizer = tokenizer_object_creation(processed_pairs[:, 0])
english_vocabulary_size = len(english_tokenizer.word_index) + 1
#english_tokenizer.word_index is a dictionary that stores the counts of each words occuring.
#print (english_tokenizer.word_index["was"])

english_max_sentence_length = max_sentence_length(processed_pairs[:, 0])
print('English Vocabulary Size: %d' % (english_vocabulary_size))
print('English Max Length: %d' % (english_max_sentence_length))


# prepare german tokenizer
german_tokenizer = tokenizer_object_creation(processed_pairs[:, 1])
german_vocabulary_size = len(german_tokenizer.word_index) + 1
#print (german_tokenizer.word_index["danke"])
german_max_sentence_length = max_sentence_length(processed_pairs[:, 1])
print('German Vocabulary Size: %d' % (german_vocabulary_size))
print('German Max Length: %d' % (german_max_sentence_length))

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [np.argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

model = load_model("model.h5")

test = input("please enter a sentence in German..")

test = data_preprocessing_for_input(test)

test_sample = encode_input_sequences(german_tokenizer, german_max_sentence_length, test)


English_translation = predict_sequence(model, english_tokenizer, test_sample)
print (English_translation)
