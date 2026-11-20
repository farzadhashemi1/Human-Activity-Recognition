# Human Activity Recognition Using Deep Learning (HAR)

A deep learning project for recognizing human physical activities from smartphone sensor data using Recurrent Neural Networks (RNNs), particularly LSTM architectures, implemented with PyTorch.

## 📌 Project Overview

Human Activity Recognition (HAR) is a classification problem that aims to identify human activities based on sensor measurements collected from wearable devices or smartphones.

In this project, we use time-series data collected from smartphone accelerometers and gyroscopes to classify six different human activities.

The main objective is to explore recurrent neural networks for sequential data classification and evaluate their ability to recognize different human activities.

## 📊 Dataset

This project uses the **UCI Human Activity Recognition Using Smartphones Dataset**.

The dataset was collected from 30 volunteers performing six different activities while carrying a smartphone on their waist.

### Activities

The model classifies the following six activities:

| Label | Activity |
|---|---|
| 0 | WALKING |
| 1 | WALKING_UPSTAIRS |
| 2 | WALKING_DOWNSTAIRS |
| 3 | SITTING |
| 4 | STANDING |
| 5 | LAYING |

### Input Features

The dataset contains nine time-series signals extracted from smartphone sensors:

**Body Acceleration:**
- body_acc_x
- body_acc_y
- body_acc_z

**Total Acceleration:**
- total_acc_x
- total_acc_y
- total_acc_z

**Body Gyroscope:**
- body_gyro_x
- body_gyro_y
- body_gyro_z

### Dataset Information

| Property | Value |
|---|---|
| Number of activities | 6 |
| Number of subjects | 30 |
| Number of input features | 9 |
| Sequence length | 128 |
| Training samples | 7,352 |
| Test samples | 2,947 |

Each sample represents a sequence of 128 time steps with 9 sensor features.

**Input shape:**

`(samples, 128, 9)`

### Dataset Download

You can download the original dataset from:

[UCI Machine Learning Repository - Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

After downloading and extracting the dataset, place the `UCI HAR Dataset` directory in your project data directory.

## 🏗️ Project Structure

```text
HAR-Activity-Recognition/
│
├── models/
│   └── best_rnn_model.pt
│
├── notebooks/
│   └── HAR_RNN_Classification.ipynb
│
├── README.md
│
├── requirements.txt
│
└── .gitignore
```

## ⚙️ Data Preprocessing

The following preprocessing steps are performed:

1. Load the nine inertial sensor signals.
2. Combine the signals into a single tensor.
3. Rearrange the tensor dimensions into the format required by recurrent neural networks.
4. Convert activity labels into zero-based class indices.
5. Separate the training, validation, and test datasets.

### Data Shape

The input data is represented as:

`(samples, sequence_length, features)`

For this project:

`(samples, 128, 9)`

The original training set is divided into training and validation subsets. The original test set is kept separate for final evaluation.

Subject-wise splitting can be used for validation to prevent subject overlap between training and validation sets.

## 🧠 Model Architecture

This project implements an LSTM-based Recurrent Neural Network using PyTorch.

### LSTM Architecture

| Layer | Configuration |
|---|---|
| Input | 9 features |
| LSTM | 2 layers |
| Hidden Size | 128 |
| Batch First | True |
| Temporal Aggregation | Mean Pooling |
| Fully Connected | 128 → 6 |
| Output | 6 activity classes |

### Model Workflow

```text
Input Sequence
(batch_size, 128, 9)
        |
        v
     LSTM Layer 1
        |
        v
     LSTM Layer 2
        |
        v
    Mean Pooling
        |
        v
 Fully Connected Layer
      (128 → 6)
        |
        v
   Classification
        |
        v
   6 Activity Classes
```

The LSTM processes the input sequence and extracts temporal representations.

Mean pooling is applied across the time dimension to aggregate information from all time steps.

Finally, a fully connected layer produces six output logits corresponding to the activity classes.

## 🚀 Training Configuration

The model is trained using the following configuration:

| Hyperparameter | Value |
|---|---|
| Model | LSTM |
| Input Size | 9 |
| Hidden Size | 128 |
| Number of Layers | 2 |
| Number of Classes | 6 |
| Batch Size | 64 |
| Loss Function | CrossEntropyLoss |
| Optimizer | SGD |
| Momentum | 0.9 |
| Learning Rate | 0.5 |
| Maximum Epochs | 50 |

The learning rate is selected through preliminary experiments.

The best model checkpoint is selected based on validation loss.

## 📈 Evaluation

The model is evaluated on the original test set, which is kept separate from training and validation.

The following evaluation metrics are used:

- Test Loss
- Accuracy
- Confusion Matrix

## 💻 Installation

### 1. Clone the repository

```bash
git clone https://github.com/farzadhashemi1/HAR-Activity-Recognition.git
```

### 2. Navigate to the project directory

```bash
cd HAR-Activity-Recognition
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the dataset

Download and extract the UCI HAR dataset using the link provided above.

### 5. Run the notebook

Open the notebook using Jupyter:

```bash
jupyter notebook
```

Navigate to:

`notebooks/HAR_RNN_Classification.ipynb`

Run the notebook cells sequentially.

## 🛠️ Technologies

- Python
- PyTorch
- NumPy
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## 🔮 Future Improvements

Potential improvements and future experiments include:

- Comparing Vanilla RNN, LSTM, and GRU architectures.
- Experimenting with different hidden sizes and network depths.
- Comparing different optimizers.
- Exploring CNN-based approaches for time-series classification.
- Investigating Attention mechanisms.
- Performing subject-wise cross-validation.
- Improving generalization to unseen subjects.

## 📚 References

- Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones.
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)

## 👨‍💻 Author

**Farzad Hashemi**

GitHub: [farzadhashemi1](https://github.com/farzadhashemi1)

---

⭐ If you find this project interesting, consider giving it a star!
