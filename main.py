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
import xgboost as xgb
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

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


# Scaling the data to be between 0 and 1
min_max_scaler = preprocessing.MinMaxScaler()
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.fit_transform(X_test)

# Let's confirm that the scaling worked as intended.
# All values should be between 0 and 1 for all variables.
X_Stats = pandas.DataFrame(X_train)
pandas.set_option('display.max_columns', None)
X_Stats.describe()


# Turning the training and testing datasets into tensors
X_train = torch.tensor(X_train)
y_train = torch.squeeze(torch.from_numpy(y_train.to_numpy()).float())
X_test = torch.tensor(X_test)
y_test = torch.squeeze(torch.from_numpy(y_test.to_numpy()).float())
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

X_train = X_train.float()
y_train = y_train.float()
X_test = X_test.float()
y_test = y_test.float()


# Initializing the neural network class
class Net(nn.Module):

  def __init__(self, n_features):
    super(Net, self).__init__()
    self.fc1 = nn.Linear(n_features, 16)
    self.fc2 = nn.Linear(16, 8)
    self.fc3 = nn.Linear(8, 4)
    self.fc4 = nn.Linear(4, 1)

# It seems that it has helped to increase the number of hidden layers and nodes per layer.

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    return torch.sigmoid(self.fc4(x))
net = Net(X_train.shape[1])

# Loss Function
criterion = nn.BCELoss()
optimizer = SGD(net.parameters(), lr=1.0)  ## here we're creating an optimizer to train the neural network.
#This learning rate seems to be working well so far

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
X_train = X_train.to(device)
y_train = y_train.to(device)

X_test = X_test.to(device)
y_test = y_test.to(device)
net = net.to(device)
criterion = criterion.to(device)


def calculate_accuracy(y_true, y_pred):
  predicted = y_pred.ge(.5).view(-1)
  return (y_true == predicted).sum().float() / len(y_true)

def round_tensor(t, decimal_places=3):
  return round(t.item(), decimal_places)

for epoch in range(1000):

    y_pred = net(X_train)
    y_pred = torch.squeeze(y_pred)
    train_loss = criterion(y_pred, y_train)
    train_acc = calculate_accuracy(y_train, y_pred)
    y_test_pred = net(X_test)
    y_test_pred = torch.squeeze(y_test_pred)
    test_loss = criterion(y_test_pred, y_test)
    test_acc = calculate_accuracy(y_test, y_test_pred)

    print(f'''    Epoch {epoch}
    Training loss: {round_tensor(train_loss)} Accuracy: {round_tensor(train_acc)}
    Testing loss: {round_tensor(test_loss)} Accuracy: {round_tensor(test_acc)}''')

# If test loss is less than 0.02, then break. That result is satisfactory.
    if test_loss < 0.02:
        print("Num steps: " + str(epoch))
        break

    optimizer.zero_grad()
    train_loss.backward()
    optimizer.step()


# Creating a function to evaluate our input
def Diabetic_Diagnosis(Pregnancies,
                       Glucose,
                       BloodPressure,
                       SkinThickness,
                       Insulin,
                       BMI,
                       DiabetesPedigreeFunction,
                       Age
                   ):
  t = torch.as_tensor([Pregnancies,
                       Glucose,
                       BloodPressure,
                       SkinThickness,
                       Insulin,
                       BMI,
                       DiabetesPedigreeFunction,
                       Age
                       ]) \
    .float() \
    .to(device)
  output = net(t)
  return output.ge(0.5).item(), output.item()

Diabetic_Diagnosis(Pregnancies=0.1, Glucose=0.1, BloodPressure=0.1, SkinThickness=0.1, Insulin=0.1, BMI=0.1,
                   DiabetesPedigreeFunction=0.1, Age=0.1)

Diabetic_Diagnosis(Pregnancies=1, Glucose=1, BloodPressure=1, SkinThickness=1, Insulin=1, BMI=1,
                   DiabetesPedigreeFunction=1, Age=1)


# Preparation for confusion matrix

# Define categories for our confusion matrix
Categories = ['Healthy','Diabetic']

# Where y_test_pred > 0.5, we categorize it as 1, or else 0.
y_test_dummy = np.where(y_test_pred > 0.5,1,0)

# Creating a confusion matrix to visualize the results.
# Model Evaluation Part 2
Confusion_Matrix = confusion_matrix(y_test, y_test_dummy)
Confusion_DF = pandas.DataFrame(Confusion_Matrix, index=Categories, columns=Categories)
sns.heatmap(Confusion_DF, annot=True, fmt='g')
plt.ylabel('Observed')
plt.xlabel('Yhat')


# Let's conduct a linear regression and evaluate the coefficients.

Reg_Out = ols("Outcome ~ Pregnancies + Glucose + BloodPressure + SkinThickness + Insulin + BMI + DiabetesPedigreeFunction + Age",
              data = Diabetes).fit()

print(Reg_Out.summary())

#                                coef    std err          t      P>|t|      [0.025      0.975]
# --------------------------------------------------------------------------------------------
# Intercept                   -0.8539      0.085     -9.989      0.000      -1.022      -0.686
# Pregnancies                  0.0206      0.005      4.014      0.000       0.011       0.031
# Glucose                      0.0059      0.001     11.493      0.000       0.005       0.007
# BloodPressure               -0.0023      0.001     -2.873      0.004      -0.004      -0.001
# SkinThickness                0.0002      0.001      0.139      0.890      -0.002       0.002
# Insulin                     -0.0002      0.000     -1.205      0.229      -0.000       0.000
# BMI                          0.0132      0.002      6.344      0.000       0.009       0.017
# DiabetesPedigreeFunction     0.1472      0.045      3.268      0.001       0.059       0.236
# Age                          0.0026      0.002      1.693      0.091      -0.000       0.006


# Let's create a quick K-nearest neighbors model and see what we get.

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import pandas
import numpy

Accuracy_Values = []

for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)

# Calculate the accuracy of the model
    if i % 2 == 0:
    print("Iteration K =",i,"Accuracy Rate=", knn.score(X_test, y_test))
    print(knn.score(X_test, y_test))
    Accuracy_Values.append([i,knn.score(X_test, y_test)])

K_Accuracy_Pair = pandas.DataFrame(Accuracy_Values)
K_Accuracy_Pair.columns=['K','Accuracy']

# Let's see the K value where the accuracy was best:

K_Accuracy_Pair[K_Accuracy_Pair['Accuracy']==max(K_Accuracy_Pair['Accuracy'])]

# The best values for K are 10, 11, 12, 13, and 21, with an accuracy rating of ~78%.
# The neural network's accuracy was around 76% to 80% depending on the epoch.


# Let's try comparing these results to a logistic regression model.

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures

Logit = LogisticRegression()

poly_accuracy = []

polynomials = range(1,10)

for poly_degree in polynomials:
    poly = PolynomialFeatures(degree = poly_degree, interaction_only=False, include_bias=False)
    X_poly = poly.fit_transform(X_train)
    X_test_poly = poly.fit_transform(X_test)
    Logit.fit(X_poly, y_train)
    y_pred = Logit.predict(X_test_poly)
    print('Polynomial Degree:',poly_degree,'Accuracy:',round(Logit.score(X_test_poly, y_test),3))
    poly_accuracy.append([poly_degree,round(Logit.score(X_test_poly, y_test),3)])

Polynomial_Accuracy = pandas.DataFrame(poly_accuracy)
Polynomial_Accuracy.columns = ['Polynomial','Accuracy']

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)

# The logistic regression's accuracy was about 75.3%.
# The corresponding polynomial value was 3, 4, 8, or 9. I would choose polynomial=3.


# Adding support for XGBoost model.

XGB = xgb.XGBClassifier(objective='binary:logistic',
                            missing=0,
                            seed=47)
XGB.fit(X_train,
        y_train,
        verbose=True,
        early_stopping_rounds=15,
        eval_metric = 'aucpr',
        eval_set=[(X_test,y_test)])

plot_confusion_matrix(XGB,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Healthy","Diabetic"])

# As with previous datasets, XGBoost is already doing a great job with almost zero tuning.

# Let's optimize the parameters - First Pass.
To_Optimize_Parameters = {
    'max_depth':[1,2,3,4,5],
    'learning_rate':[1.0,0.1,0.01,0.001],
    'gamma':[0,0.5,1.0],
    'reg_lambda':[0,1.0,10.0],
    'scale_pos_weight':[1,3,5]
}

# Now we have the following optimal parameters for XGBoost tuning:
# {'gamma': 1.0, 'learning_rate': 0.1, 'max_depth': 2, 'reg_lambda': 1.0, 'scale_pos_weight': 1}

# Second round - Not needed
#To_Optimize_Parameters = {
#    'max_depth':[2],
#    'learning_rate':[0.1],
#    'gamma':[1.0],
#    'reg_lambda':[1.0],
#    'scale_pos_weight':[1.0]
#}

# Run the following chunks for each pass (for each To_Optimize_Parameters)
optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed=47,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid = To_Optimize_Parameters,
    scoring = 'roc_auc',
    verbose = 0,
    n_jobs = 10,
    cv = 3
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=15,
                   eval_metric='auc',
                   eval_set=[(X_test,y_test)],
                   verbose=False)

# This function will give us the optimal parameters for each pass.
print(optimal_params.best_params_)

# Now that we have the optimal parameters, let's try rerunning the XGBoost algorithm.
# {'gamma': 1.0, 'learning_rate': 0.1, 'max_depth': 2, 'reg_lambda': 1.0, 'scale_pos_weight': 1}


XGB_Refined = xgb.XGBClassifier(seed = 47,
                                objective='binary:logistic',
                                gamma=1.0,
                                learn_rate=0.1,
                                max_depth=2,
                                reg_lambda=1.0,
                                scale_pos_weights=1.0,
                                subsample=0.9,
                                colsample_bytree=0.5)

XGB_Refined.fit(X_train,
                y_train,
                verbose=True,
                early_stopping_rounds=15,
                eval_metric='aucpr',
                eval_set=[(X_test,y_test)])

plot_confusion_matrix(XGB_Refined,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Healthy","Diabetic"])

# Getting a 77.2% accuracy rate with XGBoost.
# After tuning the parameters, our model is better at predicting Healthy people and worse at predicting Diabetic people.
# Before tuning the parameters, our model is better at predicting Diabetic people.

Importances = []

bst = XGB_Refined.get_booster()
for importance_type in ('weight','gain','cover','total_gain','total_cover'):
    print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))
    Importances.append(bst.get_score(importance_type=importance_type))
print(X.columns)
Importance_DF = pandas.DataFrame(Importances)
Importance_DF.columns = Diabetes.drop('Outcome',axis=1).columns

Importance_DF = Importance_DF.set_index([['Weight','Gain','Cover','Total_Gain','Total_Cover']])

