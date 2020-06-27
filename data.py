from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import torch
import pandas as pd

all_letters = string.ascii_letters + " .,;'"
n_letter = len(all_letters)

def read_csv(path):
    return pd.read_csv(path)


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )


dataframe = read_csv("gender.csv")
dataframe = dataframe[dataframe.gender != 3]

gender_dict = {}
gender = [i for i in dataframe['gender'].unique()]

for idx, row in dataframe.iterrows():
    if row['gender'] not in gender_dict.keys():
        gender_dict[row['gender']] = []
    gender_dict[row['gender']].append(unicodeToAscii(row['name']))

n_category = len(gender)

def letterToIndex(letter):
    return all_letters.index(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1,n_letter)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

def nameToTensor(name):
    tensor = torch.zeros( len(name), 1, n_letter )
    for i in range(len(name)):
        tensor[i][0][letterToIndex(name[i])] = 1
    return tensor