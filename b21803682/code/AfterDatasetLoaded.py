import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import filters
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
from collections import Counter

np.seterr(divide='ignore', invalid='ignore')
df = pd.read_csv("dataset.csv")
print("Dataset:")
print(df)
y = genfromtxt('y.csv')

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))  # return distance

# KNN Implementation
# KNN classifier
class KNN:
    def __init__(self, k=3):  # default value of k
        self.k = k  # number of nearest neighbors that we want to consider

    def fit(self, X, y):  # X is training samples, y is the training labels
        self.X_train = X  # memorize
        self.y_train = y  # memorize

    def predict(self, X):  # predict new samples
        predicted_labels = []
        for sample in X:
            predicted_labels.append(self.helper_predict(sample))  # predict all of the samples in the test sample
        return np.array(predicted_labels)  # convert to numpy array

    def helper_predict(self, sample):
        # calculate the distances
        distances = []
        for x_train in self.X_train:
            distances.append(euclidean_distance(sample, x_train))  # create distance array
        # KNN samples
        k_indices = np.argsort(distances)[:self.k]  # return the indices of smallest values according to k value
        k_nearest_neighbor_labels = []
        for index in k_indices:
            k_nearest_neighbor_labels.append(self.y_train[index])  # get the labels of the nearest neighbors
        # Voting operation (most common label)
        most_common_item = Counter(k_nearest_neighbor_labels).most_common(
            1)  # returns 1 most common item tuple(item,freq of item)
        return most_common_item[0][0]  # return the first item


# Weighted-KNN classifier
class Weighted_KNN:
    def __init__(self, k=3):
        self.k = k  # number of nearest neighbors that we want to consider

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predicted_labels = [self.helper_predict(sample) for sample in X]
        return np.array(predicted_labels)

    def helper_predict(self, x):
        # calculate the distances
        distances = []
        for x_train in self.X_train:
            distance = euclidean_distance(x, x_train)
            distances.append(distance)
        # KNN samples
        k_indices = np.argsort(distances)[:self.k]  # return the indices of smallest values according to k value
        freq_0 = 0.0  # for class 0
        freq_1 = 0.0  # for class 1
        freq_2 = 0.0  # for class 2

        for i in k_indices:
            if (self.y_train[i] == 0):
                freq_0 += 1 / euclidean_distance(x, self.X_train[i])  # increment freq_0 for every nearest class 0 point
            elif (self.y_train[i] == 1):
                freq_1 += 1 / euclidean_distance(x, self.X_train[i])  # increment freq_1 for every nearest class 1 point
            elif (self.y_train[i] == 2):
                freq_2 += 1 / euclidean_distance(x, self.X_train[i])  # increment freq_2 for every nearest class 2 point

        if (freq_0 >= freq_1):
            if (freq_0 >= freq_2):
                return 0.0
            else:
                return 2.0
        else:
            if (freq_1 >= freq_2):
                return 1.0
            else:
                return 2.0


def siplit_train_test(df):
    X = df.iloc[:, :].values  # store all Data frame to X
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.20)  # split the data into X_train and X_test, y_train, y_test
    return X, X_train, X_test, y_train, y_test


# Normalizing The Values
def normalize_values(X_train, X_test):
    X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize the pixel values to between 0 and 1
    return X_train, X_test


def calculate_accurracy(y_pred, y_test):
    acc = np.sum(y_pred == y_test) / len(y_test)
    return acc


def printConfusionMatrix(y_pred, y_test):
    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


def cross_validation(n_splits, X, y):
    kf = KFold(n_splits=n_splits, shuffle=True)
    # Calculate accurracy for each fold
    accuracies = []
    for train_index, val_index in kf.split(X):
        X_train, y_train = X[train_index], y[train_index]
        X_val, y_val = X[val_index], y[val_index]
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        acc = np.sum(y_pred == y_val) / len(y_val)  # calculate accurracy for each fold
        accuracies.append(acc)  # add each accurracy into array
    # Reporting Mean Accuracy
    mean_acc = sum(accuracies) / len(accuracies)  # calculate mean accurracy
    return mean_acc, accuracies


X, X_train, X_test, y_train, y_test = siplit_train_test(df) # split the data into X_train,X_test
X_train, X_test = normalize_values(X_train, X_test) # normalize X_train, X_test
clf = KNN(k=5)  # KNN classifier
clf.fit(X_train, y_train)  # fit data
y_pred = clf.predict(X_test)  # make a prediction
# print(y_pred)
print("Accuracy: ", calculate_accurracy(y_pred, y_test))
printConfusionMatrix(y_pred, y_test)
# K-fold cross validation
# mean_acc, accuracies = cross_validation(5, X, y)
# print("Mean accuracy: ", mean_acc) # mean accuracy after splitting folds
# print("accuracies array: ", accuracies) # accuracy array

# Error analysis
print(confusion_matrix(y_test, y_pred))  # confusion matrix
print(classification_report(y_test, y_pred))  # classification report
