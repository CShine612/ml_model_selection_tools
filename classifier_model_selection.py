import pandas as pd
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

DATA = "Class_Data.csv"
RANDOM = random.randint(0, 100)

#Data import
dataset = pd.read_csv(DATA)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Preprocessing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM)

#Scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

model_fits = {}

#Logistical Regression
logi_classifier = LogisticRegression()
logi_classifier.fit(X_train, y_train)
logi_y_pred = logi_classifier.predict(X_test)

accuracy = accuracy_score(y_test, logi_y_pred)
confusion = confusion_matrix(y_test, logi_y_pred)

model_fits["Logistical Regression"] = (accuracy, confusion)

#K Nearest Neighbours
knn_classifier = KNeighborsClassifier()
knn_classifier.fit(X_train, y_train)
knn_y_pred = knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, knn_y_pred)
confusion = confusion_matrix(y_test, knn_y_pred)

model_fits["K Nearest Neighbours"] = (accuracy, confusion)

#Support Vector Machine
svm_classifier = SVC(kernel="linear")
svm_classifier.fit(X_train, y_train)
svm_y_pred = svm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, svm_y_pred)
confusion = confusion_matrix(y_test, svm_y_pred)

model_fits["Support Vector Machine"] = (accuracy, confusion)

#Kernel SVM
ksvm_classifier = SVC(kernel="rbf")
ksvm_classifier.fit(X_train, y_train)
ksvm_y_pred = ksvm_classifier.predict(X_test)

accuracy = accuracy_score(y_test, ksvm_y_pred)
confusion = confusion_matrix(y_test, ksvm_y_pred)

model_fits["Kernal SVM"] = (accuracy, confusion)

#Naive Bayes
nb_classifier = GaussianNB()
nb_classifier.fit(X_train, y_train)
nb_y_pred = nb_classifier.predict(X_test)

accuracy = accuracy_score(y_test, nb_y_pred)
confusion = confusion_matrix(y_test, nb_y_pred)

model_fits["Naive Bayes"] = (accuracy, confusion)

#Decision Tree Classification
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
dt_y_pred = dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, dt_y_pred)
confusion = confusion_matrix(y_test, dt_y_pred)

model_fits["Decision Tree Classification"] = (accuracy, confusion)

#Random Forest Classification
rf_classifier = RandomForestClassifier()
rf_classifier.fit(X_train, y_train)
rf_y_pred = rf_classifier.predict(X_test)

accuracy = accuracy_score(y_test, rf_y_pred)
confusion = confusion_matrix(y_test, rf_y_pred)

model_fits["Random Forest Classification"] = (accuracy, confusion)

#Order models by accuracy score
model_fits = dict(sorted(model_fits.items(), key=lambda item: item[1][0], reverse=True))

i = 1
for fit in model_fits.items():
    print(f"{i} - {fit[0]} - {'{:.4f}'.format(fit[1][0])}")
    i += 1

