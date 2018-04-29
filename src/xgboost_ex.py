import pandas
import xgboost
from sklearn import datasets
from sklearn import model_selection
from sklearn.metrics import accuracy_score

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

# fit model no training data

model = xgboost.XGBClassifier()
model.fit(X_train, y_train)

print(model)

# make the predictions on the test data
y_pred = model.predict(X_test)

predictions = [round(value) for value in y_pred]
# determine the accuracy of the classifer

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
