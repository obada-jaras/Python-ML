import pandas as pd
from sklearn import svm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
from mlxtend.evaluate import bias_variance_decomp


def readFile(name):
    initial = pd.ExcelFile(name)
    file = pd.read_excel(initial)

    file.status.replace(('bad', 'good'), (0, 1), inplace=True)
    return file


def findBestCorr(file):
    plt.figure(figsize=(12, 10))
    corr = file.corr().loc[['status']].abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any((0.4 > upper[column]) & (upper[column] > -0.4))]
    file.drop(to_drop, axis=1, inplace=True)


def svmImplement(file):
    trainX = file.drop(columns=file.columns[-1], axis=1)
    trainY = file.status

    file2 = readFile("test.xlsx")
    np.intersect1d(file, file2)

    for column in file2.columns:
        if column not in file.columns:
            file2.drop(column, axis=1, inplace=True)
    testX = file2.drop(columns=file2.columns[-1], axis=1)
    testY = file2.status

    clf = svm.SVC(kernel='linear')
    clf.fit(trainX, trainY)
    y_pred = clf.predict(testX)
    print(classification_report(testY, y_pred))
    mse, bias, var = bias_variance_decomp(clf, X_train=trainX.values, y_train=trainY.values, X_test=testX.values,
                                          y_test=testY.values, loss='mse', num_rounds=200, random_seed=1)
    print("bias: " + str(bias))
    print("variance: " + str(var))


file = readFile("training.xlsx")
findBestCorr(file)
svmImplement(file)