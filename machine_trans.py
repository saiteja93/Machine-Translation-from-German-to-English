#Author: Saiteja Sirikonda
#Project : A simple Machine Translation system using Keras
#Start date: May 22nd, 2018

#Dataset taken from page www.manythings.org/anki/
#Trying to train on a Dataset which has the highest number of samples available in the page
#Working on German - English for now. Will move on to Hindi to English soon.

import re, string
import numpy as np
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
import pandas as pd

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

def tokenizer_object_creation(lines):
	tokenizer = Tokenizer()
	tokenizer.fit_on_texts(lines)
	return tokenizer

def max_sentence_length(lines):
    #returns the length of the maximum sentence in Dataset.
    return max(len(line.split()) for line in lines)


#the main aim is to come up with a Hindi-English translator, but let us start with this.
file_name = "deu.txt"
pairs = read_data(file_name)
processed_pairs = data_preprocessing(pairs)

#Test-train split.
split = 0.8
train_length = int(len(processed_pairs) * 0.8)

train, test = processed_pairs[:train_length], processed_pairs[train_length:]



# prepare english tokenizer
english_tokenizer = tokenizer_object_creation(processed_pairs[:, 0])
english_vocabulary_size = len(english_tokenizer.word_index) + 1
#english_tokenizer.word_index is a dictionary that stores the counts of each words occuring.
print (english_tokenizer.word_index["was"])

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
