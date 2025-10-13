from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
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

param_grid = {
    'C': [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
}

svm_classifier = svm.SVC(kernel='rbf', gamma='auto')

grid_search = GridSearchCV(
    svm_classifier, 
    param_grid, 
    scoring='accuracy'
)

grid_search.fit(x_train, y_train)

print("Grid Search Results:")
results = grid_search.cv_results_
for i, (params, mean_score) in enumerate(zip(
    results['params'], 
    results['mean_test_score'], 
)):
    print(f"C={params['C']},  Accuracy: {mean_score}")

print(f"Best hyperparameters: {grid_search.best_params_}")
print(f"Best accuracy from cross validation: {grid_search.best_score_}")

best_model = grid_search.best_estimator_
y_val_pred = best_model.predict(x_val)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Validation accuracy with best model: {val_accuracy}")
y_test_pred = best_model.predict(x_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"Test accuracy: {test_accuracy}")
