import pandas as pd
from sklearn.linear_model import LogisticRegression

data = pd.read_csv("data/iris.data")

model = LogisticRegression()
model.fit(data.drop("species", axis=1), data["species"])

@app.route("/predict", methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict(data)
    return jsonify({'prediction': prediction.tolist()})
