#Author: Saiteja Sirikonda
#Project : A simple Machine Translation system using Keras
#Start date: May 22nd, 2018

#Dataset taken from page www.manythings.org/anki/
#Trying to train on a Dataset which has the highest number of samples available in the page
#Working on German - English for now. Will move on to Hindi to English soon.

import re
import string
import numpy

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
            #we remove the entries with non-printable characters
            # print (entry)
            # input("removed punctuation")
            entry = "".join(filter(lambda x: x in string.printable, entry))
            # print (entry)
            # input("removed non printable characters")
            #removing numerical entries
            # temp = [word for word in entry if word.isalpha()]
            # entry = " ".join(temp)
            # print (entry)
            element.append(entry)
        cleaned.append(element)
    print (len(cleaned))
    for i in range(10):
        print (cleaned[i][0], cleaned [i][1])
    return cleaned



file_name = "deu.txt"
pairs = read_data(file_name)
processed_pairs = data_preprocessing(pairs)
