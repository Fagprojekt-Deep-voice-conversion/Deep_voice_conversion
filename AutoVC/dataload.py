import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np

path = sys.path[0]
os.chdir(path)

def DataLoad(directory):
    path = sys.path[0]
    os.chdir(path)

    dir = list(os.walk(directory))
    name = dir[0][0]

    for j, direct in enumerate(dir[1:]):

        for i in range(len(direct[2])):
            direct[2][i] = name + '/' + dir[0][1][j] + '/' + direct[2][i]

    dictionary = {dir[i][0].replace(directory + '\\', ""): dir[i][2] for i in range(1,len(dir))}
    DataFrame = pd.DataFrame.from_dict(dictionary, orient = 'index').T

    return DataFrame

def split_data(paths):
    indices = np.arange(len(paths))
    test_size = 0.1
    train_indices, test_indices = train_test_split(indices, test_size=test_size, random_state=1234)
    train_paths = list(np.array(paths)[train_indices])
    test_paths = list(np.array(paths)[test_indices])
    return train_paths, test_paths

def DataLoad2(directory, mins = None, test_size = 0.01):
	path = sys.path[0]
	os.chdir(path)

	dir = list(os.walk(directory))
	name = dir[0][0]

	for j, direct in enumerate(dir[1:]):

		for i in range(len(direct[2])):
			direct[2][i] = name + '/' + dir[0][1][j] + '/' + direct[2][i]
			


	dictionary = {dir[i][0].replace(directory + '/', ""): dir[i][2] for i in range(1,len(dir))}
	
	data_train = []
	labels_train = []

	data_test = []
	labels_test = []

	if mins is not None:
		mins = int(mins * 12)

	for key in dictionary:
		
		train, test = train_test_split(dictionary[key], test_size = test_size, random_state=1234)
		try:
			data_train.extend(train[:mins])
			labels_train.extend([key for i in train[:mins]])


		except:
			data_train.extend(train)
			labels_train.extend([key for i in train])

		data_test.extend(test)
		labels_test.extend([key for i in test])
	
	

	return (data_train, labels_train), (data_test, labels_test)

