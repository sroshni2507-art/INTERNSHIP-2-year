import pickle
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

# Load model
with open("model/svm_iris_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load iris data for feature reference
iris = load_iris()

# Sample input (can change values)
sample_input = np.array([[5.1, 3.5, 1.4, 0.2]])

# Scale input
scaler = StandardScaler()
scaler.fit(iris.data)
sample_input = scaler.transform(sample_input)

# Prediction
prediction = model.predict(sample_input)
print("Predicted Species:", iris.target_names[prediction][0])
