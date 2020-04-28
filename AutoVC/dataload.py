import os
import sys
import pandas as pd
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


def DataLoad2(directory):
    path = sys.path[0]
    os.chdir(path)

    dir = list(os.walk(directory))
    name = dir[0][0]

    for j, direct in enumerate(dir[1:]):

        for i in range(len(direct[2])):
            direct[2][i] = name + '/' + dir[0][1][j] + '/' + direct[2][i]


    dictionary = {dir[i][0].replace(directory + '\\', ""): dir[i][2] for i in range(1,len(dir))}

    data = []
    labels = []

    for key in dictionary:
        data.extend(dictionary[key])
        labels.extend([key for i in dictionary[key]])

    return data, labels

