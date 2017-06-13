import csv
import random
import numpy as np
import scipy.stats as sc


def clean_and_prepare_dataset(filename):
    lines = csv.reader(open(filename, "r"), delimiter=" ")
    dataset = list(lines)
    labels = []
    data = []

    for line in dataset:
        string_array = line[1:8]
        float_array = [float(x) for x in string_array]
        data.append(float_array)
        labels.append(line[9])

    return data, labels


def split_dataset(dataset, labels, splitRatio):
    train_size = int(len(dataset) * splitRatio)
    train_set = []
    test_set = list(dataset)
    train_set_labels = []
    test_set_labels = list(labels)

    while len(train_set) < train_size:
        index = random.randrange(len(test_set))
        train_set.append(test_set.pop(index))
        train_set_labels.append(test_set_labels.pop(index))

    return train_set, test_set, train_set_labels, test_set_labels


def separate_by_class(dataset, train_set_labels):
    separated = {}

    for train_set_line in range(len(dataset)):
        train_line = dataset[train_set_line]
        if train_set_labels[train_set_line] not in separated:
            separated[train_set_labels[train_set_line]] = []
        separated[train_set_labels[train_set_line]].append(train_line)

    return separated


def summarize(classes):
    summaries = []

    for class_line in classes:
        pair = list()
        pair.append(np.mean(class_line))
        pair.append(np.std(class_line))
        summaries.append(pair)

    return summaries


def summarize_by_class(dataset, train_set_labels):
    separated = separate_by_class(dataset, train_set_labels)
    summaries = {}

    for class_labels, instances in separated.items():
        summaries[class_labels] = summarize(instances)

    return summaries


def calculate_class_probabilities(summaries, inputVector):
    probabilities = {}

    for class_label, class_summaries in summaries.items():
        probabilities[class_label] = 1
        for i in range(len(class_summaries)):
            mean, stdev = class_summaries[i]
            probabilities[class_label] *= np.prod(sc.norm.pdf(inputVector, mean, stdev))
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculate_class_probabilities(summaries, inputVector)
    best_label, best_prob = None, -1
    for classValue, probability in probabilities.items():
        if best_label is None or (probability > best_prob and probability != float('Inf')):
            best_prob = probability
            best_label = classValue
    return best_label


def get_predictions(summaries, test_set):
    predictions = []
    for i in range(len(test_set)):
        print("Test: ", i)
        result = predict(summaries, test_set[i])
        predictions.append(result)
    return predictions


def get_accuracy(test_set_labels, predictions):
    correct = 0
    print(test_set_labels)
    print(predictions)
    for i in range(len(test_set_labels)):
        if test_set_labels[i] == predictions[i]:
            correct += 1
    return (correct / float(len(test_set_labels))) * 100.0


def naive_bayes():
    filename = 'yeast.txt'
    split_ratio = 0.9
    dataset, labels = clean_and_prepare_dataset(filename)
    training_set, test_set, train_set_labels, test_set_labels = split_dataset(dataset, labels, split_ratio)
    print('Split {0} rows into train={1} and test={2} rows'.format(len(dataset), len(training_set), len(test_set)))
    # prepare model
    summaries = summarize_by_class(training_set, train_set_labels)
    # test model

    predictions = get_predictions(summaries, test_set)
    accuracy = get_accuracy(test_set_labels, predictions)
    print('Accuracy: {0}%'.format(accuracy))
