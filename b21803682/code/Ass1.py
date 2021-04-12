import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os
import filters

## Creating DataFrame
DATADIR = "train"
CATEGORIES = ["COVID", "NORMAL", "Viral Pneumonia"]
num = 1
df_original = pd.DataFrame()
df_gabor = pd.DataFrame()
df_canny = pd.DataFrame()
y = np.array([]).astype(np.float32)
gabor_and_original_df = pd.DataFrame()
SIZE = 64
for category in CATEGORIES:
    path = os.path.join(DATADIR, category)
    class_num = CATEGORIES.index(category)
    for img in os.listdir(path):
        img_gray_scale = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE).astype(np.float32)
        img_gray_scale = cv2.resize(img_gray_scale, (SIZE, SIZE))

        img = cv2.imread(os.path.join(path, img)).astype(np.float32)
        img = cv2.resize(img, (SIZE, SIZE))

        # original pixel features
        df_original["original" + str(num)] = img_gray_scale.reshape(-1)

        # gabor process
        out = filters.Gabor_process(img)
        df_gabor["Gabor" + str(num)] = out.reshape(-1)

        # canny edge process
        out_canny = filters.Canny_edge(img)
        df_canny["Canny" + str(num)] = out_canny.reshape(-1)

        # concatenate the futures in order to increase accuracy
        result = pd.concat([df_original["original" + str(num)], df_gabor["Gabor" + str(num)]], ignore_index=True,
                           sort=False)
        gabor_and_original_df["img" + str(num)] = result
        y = np.append(y, class_num)
        num += 1

gabor_and_original_df = gabor_and_original_df.T  # transpose operation
df = gabor_and_original_df
df.to_csv('dataset.csv', index=False)

# Spliting Dataset into Train and Test Data

from sklearn.model_selection import train_test_split

X = df.iloc[:, :].values  # store all Data frame to X
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)  # split the data into X_train and X_test, y_train, y_test

# Normalizing The Values
X_train, X_test = X_train / 255.0, X_test / 255.0  # Normalize the pixel values to between 0 and 1

# KNN Implementation
from collections import Counter


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))  # return distance


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


# Weighted KNN Implementation
np.seterr(divide='ignore', invalid='ignore')


class Weighted_KNN:
    def __init__(self, k=3):
        self.k = k

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
        freq_0 = 0.0
        freq_1 = 0.0
        freq_2 = 0.0

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


clf = KNN(k=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)
acc = np.sum(y_pred == y_test) / len(y_test)
print(acc)

# Confusion Matrix
from sklearn.metrics import classification_report, confusion_matrix

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Error Analysis
error = []
# Calculating error for k values between the range of 1 and 40
for i in range(1, 40):
    knn = KNN(k=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

# K-fold Cross-Validation
from sklearn.model_selection import KFold

n_splits = 5  # decide how many folds
kf = KFold(n_splits=n_splits, shuffle=True)

# Calculate accurracy for each fold
accurracies = []
from sklearn.metrics import mean_squared_error

for train_index, val_index in kf.split(X):
    X_train, y_train = X[train_index], y[train_index]
    X_val, y_val = X[val_index], y[val_index]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_val)
    acc = np.sum(y_pred == y_val) / len(y_val)  # calculate accurracy for each fold
    accurracies.append(acc)  # add each accurracy into array

# Reporting Mean Accuracy
mean_acc = sum(accurracies) / len(accurracies)  # calculate mean accurracy
print(accurracies)
print("Mean accurracy: ", mean_acc)
