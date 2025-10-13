from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
from skimage.filters import gabor_kernel, gabor
import matplotlib.pyplot as plt
import numpy as np
import cv2

ds = fetch_openml('mnist_784', as_frame=False)
x, x_test, y, y_test = train_test_split(ds.data, ds.target, test_size=0.2, random_state=42)

unique_classes = np.unique(y)
train_data = []
train_labels = []
val_data = []
val_labels = []

np.random.seed(42)
for digit in unique_classes:
    digit_indices = np.where(y == digit)[0]
    selected_indices = np.random.choice(digit_indices, size=200, replace=False)
    
    train_indices = selected_indices[:100]
    val_indices = selected_indices[100:]
    
    train_data.append(x[train_indices])
    train_labels.append(y[train_indices])
    val_data.append(x[val_indices])
    val_labels.append(y[val_indices])

x_train = np.vstack(train_data)
y_train = np.hstack(train_labels)
x_val = np.vstack(val_data)
y_val = np.hstack(val_labels)

def downsample_images(images):
    downsampled = []
    for i in range(len(images)):
        img = images[i].reshape(28, 28).astype('uint8')
        img_resized = cv2.resize(img, (14, 14))
        downsampled.append(img_resized)
    return np.array(downsampled)

x_train_14x14 = downsample_images(x_train)
x_val_14x14 = downsample_images(x_val)

theta = np.arange(0, np.pi, np.pi/4)  
frequency = np.arange(0.05, 0.5, 0.15)  
bandwidth = np.arange(0.3, 1, 0.3)  

# plt.figure(figsize=(12, 8))
# idx = 1
# for t in theta:
#     for f in frequency:
#         for bw in bandwidth:
#             gk = gabor_kernel(frequency=f, theta=t, bandwidth=bw)
#             plt.subplot(len(theta), len(frequency)*len(bandwidth), idx)
#             plt.imshow(gk.real, cmap='gray')
#             plt.axis('off')
#             idx += 1
# plt.suptitle('Gabor Filter Bank')
# plt.tight_layout()
# plt.show()

def extract_gabor_features(images):
    n_images = len(images)
    n_filters = len(theta) * len(frequency) * len(bandwidth)
    features = np.zeros((n_images, 14 * 14 * n_filters))
    
    for i, img in enumerate(images):
        img_features = []
        
        for t in theta:
            for freq in frequency:
                for bw in bandwidth:
                    filtered_real, _ = gabor(img, frequency=freq, theta=t, bandwidth=bw)
                    img_features.append(filtered_real.flatten())
        
        features[i] = np.concatenate(img_features)
        
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{n_images} images")
    
    return features

print("Process for training extraction")
train_features = extract_gabor_features(x_train_14x14)

print("Process for extracting valuation features.")
val_features = extract_gabor_features(x_val_14x14)

classifier = svm.SVC(C=1.0, kernel='rbf', gamma='scale')
classifier.fit(train_features, y_train)

y_train_pred = classifier.predict(train_features)
train_accuracy = accuracy_score(y_train, y_train_pred)

y_val_pred = classifier.predict(val_features)
val_accuracy = accuracy_score(y_val, y_val_pred)

print(f"Results:")
print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Accuracy: {val_accuracy}")
