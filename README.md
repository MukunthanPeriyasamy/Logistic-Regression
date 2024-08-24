
# Logistic Regression Project

This project demonstrates the implementation of a Logistic Regression model for classification tasks. Below is a step-by-step guide on the process, including importing libraries, reading datasets, feature scaling, training the model, making predictions, and evaluating the results.

## Sections

1. **Importing Libraries**
2. **Reading Dataset**
3. **Feature Scaling**
4. **Training Logistic Regression**
5. **Predicting Test Result**
6. **Confusion Matrix**

### 1. Importing Libraries

First, import all the necessary libraries required for data manipulation, model building, and evaluation. Common libraries include:

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score
```

### 2. Reading Dataset

Load the dataset into a pandas DataFrame. Ensure that you have the dataset file in the same directory or provide the correct path.

```python
# Load dataset
dataset = pd.read_csv('data.csv')

# Display the first few rows
print(dataset.head())
```

### 3. Feature Scaling

Feature scaling is crucial for models that are sensitive to the scale of input features. Standardize the features using `StandardScaler`.

```python
# Separate features and target variable
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 4. Training Logistic Regression

Train the Logistic Regression model using the training dataset.

```python
# Initialize and train the model
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 5. Predicting Test Result

Use the trained model to make predictions on the test dataset.

```python
# Make predictions
y_pred = model.predict(X_test)
```

### 6. Confusion Matrix

Evaluate the performance of the model using a confusion matrix and other metrics such as accuracy.

```python
# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print('Confusion Matrix:\n', cm)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy:', accuracy)
```

## Requirements

To run this project, you need the following Python libraries:
- numpy
- pandas
- scikit-learn

You can install them using pip:

```sh
pip install numpy pandas scikit-learn
```

