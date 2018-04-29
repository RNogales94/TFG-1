import pandas
from sklearn import model_selection
from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.ensemble import AdaBoostClassifier #For Classification
from sklearn.ensemble import AdaBoostRegressor #For Regression

# import dataset
iris = datasets.load_iris()

# split data into X and y
X = iris.data
Y = iris.target

seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = model_selection.train_test_split(X,
																	Y,
																	test_size=test_size,
																	random_state=seed)

model = AdaBoostClassifier(n_estimators=100, learning_rate=1.0)
model.fit(X_train, y_train)

print(model)

# make predictions for test data
y_pred = model.predict(X_test)
predictions = [round(value) for value in y_pred]

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)

print("Accuracy: %.2f%%" % (accuracy * 100.0))
