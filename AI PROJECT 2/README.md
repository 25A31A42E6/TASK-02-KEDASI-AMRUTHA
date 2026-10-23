# Data Classification Using AI

## Iris Flower Classification

### Project Objective
To build a machine learning model that predicts the species of an Iris flower based on its measurements.

### Dataset
The Iris dataset contains 150 samples and four features:
- Sepal length
- Sepal width
- Petal length
- Petal width

The three species are Setosa, Versicolor, and Virginica.

### Technologies Used
- Python
- Scikit-learn
- Matplotlib

### Algorithm
K-Nearest Neighbors (KNN) with 3 neighbors.

StandardScaler is used for feature scaling.

### Project Workflow
1. Load the Iris dataset.
2. Split the dataset into training and testing sets.
3. Scale the features.
4. Train the KNN model.
5. Evaluate the model using accuracy, a classification report, and a confusion matrix.
6. Predict a new flower's species from user input.

### Results
The recorded model accuracy was 93%. The sample flower with measurements 5.1, 3.5, 1.4, and 0.2 cm was predicted as Setosa.

### How to Run
Install the required libraries:

`pip install scikit-learn matplotlib`

Run the Python file:

`python iris_classification.py`

### Conclusion
This project demonstrates supervised machine learning, feature scaling, classification, and model evaluation using Python.
