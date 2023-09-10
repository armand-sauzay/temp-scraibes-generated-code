import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data/iris.data")

X = data.drop("species", axis=1)
Y = data["species"]

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, Y_train)

accuracy = model.score(X_test, Y_test)
print(f"Accuracy: {accuracy:.2f}")
