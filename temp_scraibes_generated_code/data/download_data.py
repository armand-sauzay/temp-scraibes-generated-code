import requests

data_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

data = requests.get(data_url).content

with open("data/iris.data", "wb") as f:
    f.write(data)
