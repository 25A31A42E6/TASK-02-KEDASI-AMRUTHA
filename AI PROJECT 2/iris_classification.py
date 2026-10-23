
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)
import matplotlib.pyplot as plt

# 1. Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. Split the dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Create the KNN classification model
model = KNeighborsClassifier(n_neighbors=3)

# 5. Train the model
model.fit(X_train, y_train)

# 6. Predict the test data
y_pred = model.predict(X_test)

# 7. Display model accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\n=== Iris Flower Classification ===")
print("Total flowers:", len(iris.data))
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Model accuracy:", round(accuracy * 100, 2), "%")

# 8. Display classification report
print("\n=== Classification Report ===")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names,
        zero_division=0
    )
)

# 9. Display confusion matrix
cm = confusion_matrix(y_test, y_pred)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=iris.target_names
)

display.plot()
plt.title("Iris Flower Classification - Confusion Matrix")
plt.tight_layout()

# 10. Predict a new flower
print("\n=== Predict a New Flower ===")
print("Enter the flower measurements in cm:")

sepal_length = float(input("Sepal length: "))
sepal_width = float(input("Sepal width: "))
petal_length = float(input("Petal length: "))
petal_width = float(input("Petal width: "))

sample = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

# Scale the new flower measurements before prediction
sample_scaled = scaler.transform(sample)

prediction = model.predict(sample_scaled)
flower_name = iris.target_names[prediction[0]]

print("Predicted flower:", flower_name)

# 11. Show the confusion matrix graph
plt.show()
