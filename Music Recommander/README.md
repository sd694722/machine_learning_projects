# Music Genre Classifier

## Overview

This project involves building a machine learning model to classify music genres based on age and gender using a Decision Tree Classifier. The model is trained on a dataset and can predict the genre of music for new input data.

## Table of Contents

- [Project Description](#project-description)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Saving and Loading the Model](#saving-and-loading-the-model)
- [Contributing](#contributing)
- [License](#license)

## Project Description

The Music Genre Classifier project uses a Decision Tree Classifier to predict music genres. The dataset contains features such as age and gender and labels representing different music genres. The classifier is trained on a portion of the dataset and tested on another to evaluate its performance.

## Installation

To get started with this project, you'll need to have Python installed. You can then install the required packages using `pip`.

```bash
pip install pandas scikit-learn joblib


## Usage

1. **Load the Dataset:**

   To load the dataset into a Pandas DataFrame, use the following code:

   ```python
   import pandas as pd
   
   # Load the dataset
   df = pd.read_csv('music.csv')
