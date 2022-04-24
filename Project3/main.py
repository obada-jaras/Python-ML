import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from mlxtend.evaluate import bias_variance_decomp


def read_file(file_name):
    fileR = pd.read_csv(file_name)

    # Drop the first column (ID) as it's not required
    fileR = fileR.drop(['ID'], axis=1)

    return fileR


def reformat_data(data):
    labels = data['label'].values
    dataset = data.drop(['label'], axis=1)

    return labels, dataset


def split_data(data):
    # splitting data randomly to training and testing data
    train, test = train_test_split(data, test_size=0.2)
    return train, test


def c45_prediction(labels, dataset):
    clf = DecisionTreeClassifier()
    clf = clf.fit(dataset, labels)

    return clf


def rf_prediction(labels, dataset):
    clf = RandomForestClassifier()
    clf = clf.fit(dataset, labels)

    return clf


def main():
    file = read_file("Dataset.csv")

    training_data, testing_data = split_data(file)

    # label is the result column (to be predicted), dataset is the data without the label column (data affects on the result) 
    training_labels, training_dataset = reformat_data(training_data)
    testing_labels, testing_dataset = reformat_data(testing_data)

    # clf is the classification model
    c45_clf = c45_prediction(training_labels, training_dataset)
    rf_clf = rf_prediction(training_labels, training_dataset)

    # predicted labels for all testing data
    label_pred_c45 = c45_clf.predict(testing_dataset)
    label_pred_rf = rf_clf.predict(testing_dataset)

    # define reports that contain: precision, recall, f1-score, accuracy
    c45_report = classification_report(testing_labels, label_pred_c45)
    rf_report = classification_report(testing_labels, label_pred_rf)

    # finding mse, bias, var for C4.5 model and Random Forest model
    c45_mse, c45_bias, c45_var = bias_variance_decomp(c45_clf, X_train=training_dataset.values, y_train=training_labels,
                                                      X_test=testing_dataset.values, y_test=testing_labels, loss='mse',
                                                      num_rounds=200, random_seed=1)
    rf_mse, rf_bias, rf_var = bias_variance_decomp(rf_clf, X_train=training_dataset.values, y_train=training_labels,
                                                   X_test=testing_dataset.values, y_test=testing_labels, loss='mse',
                                                   num_rounds=200, random_seed=1)

    # printing all the results
    print("Results For Decision Tree C4.5 model: ")
    print(c45_report)
    print("MSE = " + str(c45_mse))
    print("Bias = " + str(c45_bias))
    print("Variance = " + str(c45_var))

    print("\n###############\n")

    print("Results For Random Forest model: ")
    print(rf_report)
    print("MSE = " + str(rf_mse))
    print("Bias = " + str(rf_bias))
    print("Variance = " + str(rf_var))


if __name__ == "__main__":
    main()