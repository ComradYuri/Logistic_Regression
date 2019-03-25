import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Setting up pandas so that it displays all columns instead of collapsing them
desired_width = 400
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns', 15)

# 1 Load the passenger data
passengers = pd.read_csv('passengers.csv')
print("1------------------------------------")
print(passengers.head())

# 2 Update sex column to numerical
passengers.Sex = passengers.Sex.map({"female": 1, "male": 0})
print("2------------------------------------")
print(passengers.head())

# 3 Fill the nan values in the age column
passengers.Age.fillna(value=passengers.Age.mean(), inplace=True)
print("3------------------------------------")
print(passengers.Age.values[:10])

# 4 Create a first class column
passengers["FirstClass"] = passengers.Pclass.apply(lambda x: 1 if x == 1 else 0)

# 5 Create a second class column
passengers["SecondClass"] = passengers.Pclass.apply(lambda x: 1 if x == 2 else 0)
print("5------------------------------------")
print(passengers.head(10))

# 6 Select the desired features
features = passengers[["Sex", 'Age', 'FirstClass', 'SecondClass']].astype(float)
survival = passengers.Survived
print("6------------------------------------")
print(features.head())
print(survival.head())

# 7 Perform train, test, split
x_train, x_test, y_train, y_test = train_test_split(features, survival, test_size=.2)

# 8 Scale the feature data so it has mean = 0 and standard deviation = 1
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# 9 Create and train the model
model = LogisticRegression(solver='lbfgs')
model.fit(x_train, y_train)

# 10 Score the model on the train data
print("10------------------------------------")
print(model.score(x_train, y_train))

# 11 Score the model on the test data
print("11------------------------------------")
print(model.score(x_test, y_test))

# 12 Analyze the coefficients
print("12-----------------------------------")
print(np.array(["Sex", "Age", "FirstClass", "SecondClass"]))
print(model.coef_)
features = model.coef_.tolist()[0]

ax = plt.subplot()
ax.set_xticks(range(len(features)))
ax.set_xticklabels(["Is Female", "Age", "Is First Class", "Is Second Class"])
plt.bar(range(len(features)), features)
plt.ylabel("Coefficient")
plt.title("12 Effect on survivability during the Titanic sinking")
plt.show()
plt.close("all")

# 13 Sample passenger features
Jack = np.array([0., 20., 0., 0.])
Rose = np.array([1., 17., 1., 0.])
You = np.array([0., 25., 0., 0.])

# 14 Combine passenger arrays
sample_passengers = np.array([Jack, Rose, You])
print("14-----------------------------------")
print(sample_passengers)

# 15 Scale the sample passenger features
sample_passengers = scaler.transform(sample_passengers)
print("15-----------------------------------")
print(sample_passengers)

# 16 Make survival predictions!
print("16-----------------------------------")
print("[Pdie]        [Psurvive]")
print(model.predict_proba(sample_passengers))
print(model.predict(sample_passengers))
