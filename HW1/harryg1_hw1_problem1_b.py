from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
import cv2


ds = fetch_openml('mnist_784', as_frame=False)
x, x_test, y, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)

unique_classes = np.unique(y)
balanced_data = []
balanced_labels = []

np.random.seed(42)
for digit in unique_classes:
    digit_indices = np.where(y == digit)[0]
    selected_indices = np.random.choice(digit_indices, size=1000, replace=False)
    balanced_data.append(x[selected_indices])
    balanced_labels.append(y[selected_indices])

x_subsampled = np.vstack(balanced_data)
y_subsampled = np.hstack(balanced_labels)

x_downsampled = []
for i in range(len(x_subsampled)):
    img = x_subsampled[i].reshape(28, 28).astype('uint8')
    img_resized = cv2.resize(img, (14, 14))
    x_downsampled.append(img_resized.flatten())
x_subsampled = np.array(x_downsampled)

x_test_downsampled = []
for i in range(len(x_test)):
    img = x_test[i].reshape(28, 28).astype('uint8')
    img_resized = cv2.resize(img, (14, 14))
    x_test_downsampled.append(img_resized.flatten())
x_test = np.array(x_test_downsampled)

x_train, x_val, y_train, y_val = train_test_split(x_subsampled, y_subsampled, test_size=0.2, random_state=42)

classifier = svm.SVC(C=1.0, kernel='rbf', gamma='auto')
classifier.fit(x_train, y_train)
y_val_pred = classifier.predict(x_val)

# Val error 
val_accuracy = accuracy_score(y_val, y_val_pred)
val_error = 1 - val_accuracy
print(f"Validation Error: {val_error}")

# Support ratio
total_support = classifier.n_support_.sum()
support_ratio = total_support / len(x_train)
print(f"Support samples ratio: {support_ratio}")

# Classificaiton error
y_test_pred = classifier.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_error = 1 - test_accuracy
print(f"Classification Error: {test_error}")

# Confusion 
cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(cm)