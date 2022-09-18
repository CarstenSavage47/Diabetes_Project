import torch  # torch provides basic functions, from setting a random seed (for reproducability) to creating tensors.
import torch.nn as nn  # torch.nn allows us to create a neural network.
import torch.nn.functional as F  # nn.functional give us access to the activation and loss functions.
from torch.optim import SGD  # optim contains many optimizers. Here, we're using SGD, stochastic gradient descent.
import matplotlib.pyplot as plt  ## matplotlib allows us to draw graphs.
import seaborn as sns  ## seaborn makes it easier to draw nice-looking graphs.
import os
from tqdm import tqdm
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import preprocessing
from statsmodels.formula.api import ols

import pandas
import numpy as np

Diabetes = pandas.read_csv('/Users/carstenjuliansavage/Desktop/R Working Directory/diabetes.csv')

pandas.set_option('display.max_columns', None)

Diabetes.describe()

X = (Diabetes
 .drop(['Outcome'],axis=1)
)

y = Diabetes['Outcome']

# Split dataframe into training and testing data. Remember to set a seed and stratify.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=47, stratify=y)

# Let's check to make sure that y_train and y_test are successfully stratified. 
y_train.agg({'mean'})
y_test.agg({'mean'})


