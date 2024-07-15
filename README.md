# Image Classification with SVM

This repository contains a Jupyter Notebook that demonstrates an image classification pipeline using Support Vector Machine (SVM) with Histogram of Oriented Gradients (HOG) features. The notebook is designed to be run in Google Colab and uses a dataset stored on Google Drive.

## Table of Contents

- [Introduction](#introduction)
- [Dependencies](#dependencies)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Introduction

This project demonstrates the process of building an image classification model using SVM. It includes steps for loading data, preprocessing images, extracting features using HOG, training the SVM model, and evaluating its performance on a validation set.

## Dependencies

The following libraries are required to run the notebook:

- `numpy`
- `pandas`
- `scikit-learn`
- `opencv-python`
- `scikit-image`
- `google-colab` (for Google Drive integration)

You can install these dependencies using pip:

```bash
pip install numpy pandas scikit-learn opencv-python scikit-image google-colab
```

## Usage

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/your-repo-name.git
    cd your-repo-name
    ```

2. **Open the notebook in Jupyter or Google Colab:**

    - If using Jupyter Notebook:
    
        ```bash
        jupyter notebook PRODIGY_ML_03.ipynb
        ```

    - If using Google Colab:
    
        Upload the `PRODIGY_ML_03.ipynb` notebook to your Google Drive and open it with Google Colab.

3. **Run the cells in the notebook sequentially.** Make sure to mount your Google Drive and adjust the paths to your dataset accordingly.

## Dataset

The dataset used in this project should be a zipped file containing training and validation images, along with their respective CSV files containing labels. The dataset structure should be as follows:

```
archive.zip
│
└───train/
│   └───cat/
│   └───dog/
│
└───val/
│   └───cat/
│   └───dog/
│
└───train.csv
└───val.csv
```

Update the paths in the notebook to point to the location of your dataset in Google Drive.

## Model Training and Evaluation

The notebook includes the following steps:

1. **Mount Google Drive** to access the dataset.
2. **Extract the dataset** from a ZIP file.
3. **Load the CSV files** containing image labels.
4. **Preprocess the images** and extract HOG features.
5. **Split the data** into training and validation sets.
6. **Train an SVM model** with the HOG features.
7. **Evaluate the model** on the validation set.

## Results

The notebook prints the accuracy and classification report for the validation set. Example output:

```
Validation Accuracy: 0.63
              precision    recall  f1-score   support

         cat       0.42      0.21      0.28        24
         dog       0.67      0.85      0.75        46

    accuracy                           0.63        70
   macro avg       0.54      0.53      0.51        70
weighted avg       0.58      0.63      0.59        70
```
