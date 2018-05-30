#Author: Saiteja Sirikonda
#Project : A simple Machine Translation system using Keras
#Start date: May 22nd, 2018

#Dataset taken from page www.manythings.org/anki/
#Trying to train on a Dataset which has the highest number of samples available in the page
#Working on German - English for now. Will move on to Hindi to English soon.

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

# def data_preprocessing(pairs):
#     #removing punctuation, converting them to lowercase, removing the non-printable characters and removing the with numbers in them.
#     cleaned = []
#     for i in pairs:
#         element = []
#         for entry in i:
#             #we start, by removing punctuation and turning to lowercase
#             entry = re.sub(r'[^\w\s]',"",entry.lower())
#             # we remove the entries with non-printable characters
#             # print (entry)
#             # input("removed punctuation")
#             entry = "".join(filter(lambda x: x in string.printable, entry))
#             # print (entry)
#             # input("removed non printable characters")
#             # removing numerical entries
#             # temp = [word for word in entry if word.isalpha()]
#             # entry = " ".join(temp)
#             # print (entry)
#             element.append(entry)
#         cleaned.append(element)
#     #print (len(cleaned))
#     # for i in range(10):
#     #     print (cleaned[i][0], cleaned [i][1])
#     return np.array(cleaned)

def data_preprocessing(lines):
	cleaned = list()
	# prepare regex for char filtering
	re_print = re.compile('[^%s]' % re.escape(string.printable))
	# prepare translation table for removing punctuation
	table = str.maketrans('', '', string.punctuation)
	for pair in lines:
		clean_pair = list()
		for line in pair:
			# normalize unicode characters
			line = normalize('NFD', line).encode('ascii', 'ignore')
			line = line.decode('UTF-8')
			# tokenize on white space
			line = line.split()
			# convert to lowercase
			line = [word.lower() for word in line]
			# remove punctuation from each token
			line = [word.translate(table) for word in line]
			# remove non-printable chars form each token
			line = [re_print.sub('', w) for w in line]
			# remove tokens with numbers in them
			line = [word for word in line if word.isalpha()]
			# store as string
			clean_pair.append(' '.join(line))
		cleaned.append(clean_pair)
	return np.array(cleaned)


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

def Model_specifications(source, target, source_max, target_max, dimension):
	model = Sequential()
	model.add(Embedding(source, dimension, input_length=source_max, mask_zero=True))
	model.add(LSTM(dimension))
	model.add(RepeatVector(target_max))
	model.add(LSTM(dimension, return_sequences=True))
	model.add(TimeDistributed(Dense(target, activation='softmax')))
	return model

#the main aim is to come up with a Hindi-English translator, but let us start with this.
file_name = "deu.txt"
pairs = read_data(file_name)
processed_pairs = data_preprocessing(pairs)
processed_pairs = processed_pairs[:10000]

#Test-train split.
split = 0.8
train_length = int(len(processed_pairs) * 0.8)

train, test = processed_pairs[:train_length], processed_pairs[train_length:]



# prepare english tokenizer
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


trainX = encode_input_sequences(german_tokenizer, german_max_sentence_length, train[:, 1])
trainY = encode_input_sequences(english_tokenizer, english_max_sentence_length, train[:, 0])
trainY = encode_to_onehot(trainY, english_vocabulary_size)
# prepare validation data
testX = encode_input_sequences(german_tokenizer, german_max_sentence_length, test[:, 1])
testY = encode_input_sequences(english_tokenizer, english_max_sentence_length, test[:, 0])
testY = encode_to_onehot(testY, english_vocabulary_size)


model = Model_specifications(german_vocabulary_size, english_vocabulary_size, german_max_sentence_length, english_max_sentence_length, 512)
#categorical_crossentropy, since we framed our problem as a Multiclass classification problem.
model.compile(optimizer='adam', loss='categorical_crossentropy')
# prints the summary of the model, which basically prints the configuration of the model.
print(model.summary())
# fit model
filename = 'model.h5'
checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=2, save_best_only=True, mode='min')
model.fit(trainX, trainY, epochs=30, batch_size=8, validation_data=(testX, testY), callbacks=[checkpoint], verbose=2)
#### END OF TRAINING###

#We try to evaluate the best model, which is stored in model.h5 file. We print translations and evaluate the BLEU scoresself.

def word_for_id(integer, tokenizer):
	for word, index in tokenizer.word_index.items():
		if index == integer:
			return word
	return None

# generate target given source sequence
def predict_sequence(model, tokenizer, source):
	prediction = model.predict(source, verbose=0)[0]
	integers = [argmax(vector) for vector in prediction]
	target = list()
	for i in integers:
		word = word_for_id(i, tokenizer)
		if word is None:
			break
		target.append(word)
	return ' '.join(target)

# evaluate the skill of the model
def evaluate_model(model, tokenizer, sources, raw_dataset):
	actual, predicted = list(), list()
	for i, source in enumerate(sources):
		# translate encoded source text
		source = source.reshape((1, source.shape[0]))
		translation = predict_sequence(model, eng_tokenizer, source)
		raw_target, raw_src = raw_dataset[i]
		if i < 10:
			print('src=[%s], target=[%s], predicted=[%s]' % (raw_src, raw_target, translation))
		actual.append(raw_target.split())
		predicted.append(translation.split())
	# calculate BLEU score
	print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
	print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
	print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
	print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))

model = load_model("model.h5")

evaluate_model(model, english_tokenizer, testX, test)
