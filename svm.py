# (C) Ritvik Chitram, March 29th 2021
# Hard-boundary linear classifier implemented using a Support Vector Machine imported from scikit-learn.
# This classifier trains on past patient data and classifies whether future patients are likely to have a heart attack.
# NOTE: Patients of group 0 are those who had heart attacks, and patients of group 1 are those who did not have attacks.

import pandas as pd
from sklearn.utils import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

data_array = pd.read_excel('data.xlsx')
data_array = shuffle(data_array)

patient_data = data_array.drop('GROUP', axis=1)
attacked = data_array['GROUP']

# These print statements only serve to preview the format of the pandas dataframes
# print(patient_data.head())
# print(attacked.head())

# vary the test size in proportion to the dataset size
train_data, test_data, train_labels, test_labels = train_test_split(patient_data, attacked, test_size=0.20)

# we use a linear kernel as there is not too much overlap in this scenario
sv_classifier = SVC(kernel='linear')
sv_classifier.fit(train_data, train_labels)

predictions = sv_classifier.predict(test_data)

# we check the accuracy of our SVM on the test data
print("Confusion Matrix:\n", confusion_matrix(test_labels, predictions))
print("Classification Report:\n", classification_report(test_labels, predictions))
