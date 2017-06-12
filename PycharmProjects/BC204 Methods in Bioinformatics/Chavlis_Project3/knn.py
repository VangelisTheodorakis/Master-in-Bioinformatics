import numpy as np
import random
import operator
from scipy.spatial import distance

def clean_and_prepare_dataset(filename, split, trainingSet=[], testSet=[], train_labels=[], test_labels=[]):
        classes = np.genfromtxt(filename, dtype='U', usecols=[-1])
        lines = np.genfromtxt(filename, usecols=[1,2,3,4,5,6,7,8])
        dataset = list(lines)
        dummy = []

        for x in range(len(dataset)):
            for y in range(8):
                dataset[x][y] = float(dataset[x][y])
            if random.random() < split:
                #print(dataset[x])
                train_labels.append(str(classes[x]))
                trainingSet.append(dataset[x])
            else:
                test_labels.append(str(classes[x]))
                testSet.append(dataset[x])
            dummy.append(dataset[x])
        return dummy



def get_neighbors(training_set, test_line, k):
    distances = []

    for x in range(len(training_set)):
        dist = distance.euclidean(test_line, training_set[x])
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []

    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def find_array(train, array):
    for x in range(len(train)):
        if np.array_equal(train[x], array):
            return x

def get_closest_neighbor(neighbors, train_set, train_labels):
    classVotes = {}
    for x in neighbors:
        index = find_array(train_set,x)
        response = train_labels[index]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1

    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def get_accuracy(testSet, test_labels, predictions):
    correct = 0
    for x in range(len(testSet)):
        if test_labels[x] == predictions[x]:
            correct += 1
    return (correct / float(len(testSet))) * 100.0


def knn():
    # prepare data
    train_set = []
    testSet = []
    train_labels = []
    test_labels = []
    split = 0.67
    dummy = clean_and_prepare_dataset('yeast.txt', split, train_set, testSet, train_labels, test_labels)
    print('Train set: ' + repr(len(train_set)))
    print('Test set: ' + repr(len(testSet)))
    print('TrainLabels set: ' + repr(len(train_labels)))
    print('Test Labels set: ' + repr(len(test_labels)))
    predictions = []
    k = 3

    for x in range(len(testSet)):
        neighbors = get_neighbors(train_set, testSet[x], k)
        result = get_closest_neighbor(neighbors, train_set, train_labels)
        predictions.append(result)
        print('> predicted=' + repr(result) + ', actual=' + repr(test_labels[x]))
    accuracy = get_accuracy(testSet, test_labels, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')