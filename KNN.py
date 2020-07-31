import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import linear_model, preprocessing

data = pd.read_csv("car.data")
print(data.head())


# TRANSFORMING CLASSIFICATION DESCRIPTIONS INTO INTEGER VALUES-------------------------------------------------------


# Initialize object that lets us transfer classification descriptions into integer values
le = preprocessing.LabelEncoder()

# Take each column of the panda data frame, turn it into an array, to be utilized by
# Preprocessing (as it only takes arrays), that transforms the classifications into integers
# Return is a numpy array
buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
lug_boot = le.fit_transform(list(data["lug_boot"]))
safety = le.fit_transform(list(data["safety"]))
cls = le.fit_transform(list(data["class"]))

predict = "class"

# The zip command combines the given arrays into an array of tuples
x = list(zip(buying, maint, door, persons, lug_boot, safety))
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size = 0.1)

# Initialize KNN model, with parameters being k, the # of neighbors to compare to
# Conclude the classification. We initially started with n_neighbors = 5, then
# Increased/Decreased the parameter until higher accuracy was achieved
model = KNeighborsClassifier(n_neighbors=9)

model.fit(x_train, y_train)

acc = model.score(x_test, y_test)

print(acc)

predicted = model.predict(x_test)

# Initialize possible outputs of prediction column class, as we previously turned them into integers
# unacc: unacceptable, acc: acceptable, good: good, vgood: very good
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(predicted)):
    # predicted[x] and y_test[x] will output integers, corresponding to names, functioning as the index
    print("Predicted: ", names[predicted[x]], "Data: ", x_test[x], "Actual: ", names[y_test[x]])

# We need not save KNN models, as every time it makes a prediction, it must traverse and calculate all distances
# Between points, so it would be very time/space heavy
