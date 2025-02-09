# Thyroid Cancer Detection & ASL Detection using CNN

## 📌 Repository Overview
This repository contains two projects:
1. **Thyroid Cancer Detection** - A machine learning-based approach to detect thyroid cancer using multiple classification algorithms.
2. **ASL Detection using CNN** - A deep learning model for recognizing American Sign Language (ASL) gestures using Convolutional Neural Networks (CNNs).

---

## 🏥 Thyroid Cancer Detection
### 📖 Overview
The goal of this project is to classify whether a patient has thyroid cancer based on given features.

### 🔬 Models Used
- Logistic Regression (LR)
- Decision Tree (DT)
- Naïve Bayes (NB)
- k-Nearest Neighbors (KNN)
- **Random Forest (RF) (Best Model)**

### 🔍 Experiment Setup
1. **Without Feature Selection**: All models are trained on the full dataset.
2. **With Feature Selection**: Best features are selected to improve model performance.
3. **Best Model**: Random Forest performed best in both settings.

### 📊 Results
- **Random Forest** achieved the highest accuracy in both feature selection and non-feature selection scenarios.

---

## 👐 ASL Detection using CNN
### 📖 Overview
This project aims to classify American Sign Language (ASL) gestures using a Convolutional Neural Network (CNN).

### 🏗 Model Architecture
```python
model = Sequential()

model.add(Conv2D(64, (3, 3), padding='same', input_shape=(32, 32, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Dropout(0.2))

model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(1024, activation='relu'))
model.add(Dense(classes, activation='softmax'))
```

### 📊 Results
- The model is trained on an ASL dataset to classify different hand signs accurately.
- Uses Batch Normalization and Dropout for better generalization.
- Training Accuracy->93.17%
- Testing Accuracy-> 93.86%
---


