# 🔍 SmartVision – Industrial Image Classification System

SmartVision is an end-to-end Computer Vision application that classifies industrial scene images using a custom Convolutional Neural Network (CNN). The project combines TensorFlow, OpenCV, and Streamlit to provide real-time image classification with confidence scores and an interactive web interface.

---

## 📌 Features

- Custom CNN built using TensorFlow & Keras
- Image preprocessing using OpenCV
- Real-time image classification
- Confidence score prediction
- Top-3 class probabilities
- Interactive Streamlit web application
- Accuracy & Loss visualization
- Confusion Matrix generation
- Model checkpointing with Early Stopping

---

## 🛠 Tech Stack

- Python
- TensorFlow
- Keras
- OpenCV
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit

---

## 📂 Dataset

Intel Image Classification Dataset

Classes:

- Buildings
- Forest
- Glacier
- Mountain
- Sea
- Street

Dataset contains over 17,000 labeled images across six scene categories.

---

## 🧠 CNN Architecture

Input Image (150 × 150)

↓

Conv2D (16 Filters)

↓

Batch Normalization

↓

MaxPooling

↓

Conv2D (32 Filters)

↓

Batch Normalization

↓

MaxPooling

↓

Conv2D (64 Filters)

↓

Batch Normalization

↓

MaxPooling

↓

Conv2D (128 Filters)

↓

Batch Normalization

↓

MaxPooling

↓

GlobalAveragePooling

↓

Dense (128)

↓

Dropout

↓

Softmax (6 Classes)

---

## 📊 Model Performance

Training Accuracy

- 75.64%

Validation Accuracy

- 68.47%

Loss Function

- Categorical Crossentropy

Optimizer

- Adam

---

## 📁 Project Structure

```
SmartVision/
│
├── app.py
├── train.py
├── predict.py
├── model.py
├── utils.py
├── requirements.txt
│
├── dataset/
│   ├── seg_train/
│   └── seg_test/
│
├── models/
│   ├── cnn_model.keras
│   └── classes.json
│
├── outputs/
│   ├── accuracy.png
│   ├── loss.png
│   └── confusion_matrix.png
│
└── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/yourusername/SmartVision.git

cd SmartVision
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Train the Model

```bash
python train.py
```

The training script automatically:

- Loads the dataset
- Trains the CNN
- Saves the trained model
- Generates accuracy & loss graphs
- Creates the confusion matrix

---

## ▶️ Run the Application

```bash
streamlit run app.py
```

Upload an image and SmartVision will:

- Preprocess the image
- Predict its class
- Display confidence score
- Show Top-3 predictions
- Visualize prediction probabilities

---

## 🖼 Application Workflow

Image Upload

↓

OpenCV Image Preprocessing

↓

CNN Prediction

↓

Confidence Calculation

↓

Top-3 Predictions

↓

Interactive Streamlit Dashboard

---

## 💡 Future Improvements

- Transfer Learning using EfficientNet or MobileNetV2
- Object Detection using YOLO
- Industrial Defect Detection
- Grad-CAM Visualization
- Model Quantization
- ONNX Deployment
- Edge Device Inference

---

## 👨‍💻 Author

Ayush Kumar Gupta

GitHub:
https://github.com/AyushGupta74

Email:
connecttoayushg@gmail.com

## 📸 Screenshots

### Home Page

![Home](assets/home.png)

### Prediction

![Prediction](assets/prediction.png)

### Accuracy Curve

![Accuracy](assets/accuracy.png)

### Loss Curve

![Loss](assets/loss.png)

### Confusion Matrix

![Confusion Matrix](assets/confusion_matrix.png)