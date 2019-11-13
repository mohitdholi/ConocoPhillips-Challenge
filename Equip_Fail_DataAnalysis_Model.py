# Importing libraries
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.feature_selection import RFE
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.impute import SimpleImputer
import statsmodels.api as sm
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.metrics import f1_score
import seaborn as sns
from imblearn.over_sampling import (SMOTE, BorderlineSMOTE, SVMSMOTE)
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from tensorflow.python.keras.layers import Dense,Dropout
from tensorflow.python.keras.models import Sequential
from keras.utils import plot_model
###############################################################################
# Reading data
original_df = pd.read_csv("equip_failures_training_set.csv")
# Check the variable names and the data type
original_df.columns
# Descriptive statistics
original_df.describe()
# Since only three variables are shown that means other variables are in object
# type and will need conversion for numerical calculation.
# #############################################################################
# The data needs to be checked for any missing values. First lets gather how
# many missng values are present in the data.
missing_total = original_df.isnull().sum().sort_values(ascending=False)
missing_percent = ((original_df.isnull().sum()/original_df.isnull().count()).
                   sort_values(ascending=False))
missing_data = pd.concat([missing_total, missing_percent], axis=1,
                         keys=['Missing Total', 'Missing Percent'])
missing_data.head()
# We found no missing values in the data.
###############################################################################
# Looking at the csv file shows that the data many 'na' values. With such
# a number of rows and columns it is difficult to locate and get a sense of how
# many of these 'missing'na' values are present.
# These 'na' values are in string format and thus need to be converted to Numpy
# NaN for imputation purpose.
# Replacing string 'na' by numpy NaN values
nareplaced_df = original_df.replace('na', np.NaN)
# Lets gather how many NaN values are present in our dataset.
nan_total = nareplaced_df.isna().sum().sort_values(ascending=False)
nan_percent = ((nareplaced_df.isna().sum()/nareplaced_df.isna().count()).
               sort_values(ascending=False))
nan_data = pd.concat([nan_total, nan_percent], axis=1,
                     keys=['NaN Total', 'NaN Percent'])
nan_data.head(50)
# There are a lot of nan values in some of the variables of the dataset.
###############################################################################
# Convert the data type of the variables to float
float_df = nareplaced_df.astype(float)
float_df.describe()
###############################################################################
# IMPUTATION
# There are many things that can be done for missing/nan values.
#   1. Drop the whole row of observation containing any missing/nan value(s).
#   2. Replace the missing/nan values with mean of the variable.
#   3. Replace the missing/nan values with median of the variable.
#   4. Replace the missing/nan values with the most frequent value.
#   5. Replace the missing/nan values a constant value.
#   6. Replace the missing/nan value using multivariate fitting algorithm.
# Essentially, it is a trial-n-error method where different imputation schemes
# must be judged against each other by comparing accuracy of our model.
# First scheme of imputation
# Impute the remaining variables with their mean
# Using Scikit simple imputer: creates a Numpy array which needs to be
# converted to a dataframe
# imp = SimpleImputer(missing_values=np.nan, strategy='mean')
# imputed_array = imp.fit_transform(temp_df)
#
# Using fillna() function creates a dataframe
imputed_df = float_df.fillna(float_df.mean())
###############################################################################
# SPLITTING THE DATA INTO TRAINING AND TESTING DATASETS
# Splitting the data in training and testing dataset
np.random.seed(1910) 
df = imputed_df.sample(frac=1).reset_index(drop=True)
train, test = train_test_split(df, test_size=0.2)
imputed_df = train
imputed_df_test = test
###############################################################################
# NORMALIZATION
# If the range of dataset values vary a lot, normalization of all the variables
# must be carried out. In a regression problem, dependent and independent vari-
# ables are normalized while in categorical analysis only independent variables
# are normalized.
#
# Separating independent and dependent variables
datacols = list(imputed_df)
y = imputed_df[datacols[1]]
x = imputed_df[datacols[2:]]
count_no_failure = len(y[y == 0])
count_failure = len(y[y == 1])
y = pd.DataFrame(y)
scaled_data = pd.DataFrame()
lcol = len(x.columns)
for i in range(lcol):
    data = x[x.columns[i]]
    data = data.values.astype(float)
    data = data.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(data)
    testdf = pd.DataFrame(scaled_array, columns=[x.columns[i]])
    scaled_data = pd.concat([scaled_data, testdf.reindex(testdf.index)],
                            axis=1)
x_scaled = scaled_data
x_scaled.describe()
###############################################################################
# CLASS IMBALANCES
# The plot below shows the occurrences of classes in the dependent variable y
sns.countplot(y['target'])
# The variables count_no_failure and count_failure differ significantly meaning
# that just answering that there will be no failure always, we can achieve a
# very high accuracy ~ 98%. This is due to class imbalances. There are too many
# of one type of classes. In this case, no failure is prominent.
# In this situation, the predictive model developed using conventional machine
# learning algorithms could be biased and inaccurate.
# There are many techniques that can be used to resample the data:
# 1. Random under-sampling
# 2. Random over-sampling
# 3. Cluster-based over sampling
# 4. Synthetic Minority Over-sampling Technique (SMOTE)
# 5. Modified SMOTE
# 6. Ensemble techniques:
#    1. Bagging based
#    2. Boosting based
#    3. Adaptive Boosting (Ada Boosting)
#    4. Gradient Tree Boosting
#    5. XG Boost
# First using SMOTE for removing imbalance
x_scaled_array = x_scaled.to_numpy()
y_array = y.to_numpy()
smote = SMOTE(random_state = 0)
x_smote, y_smote = smote.fit_sample(x_scaled_array, y_array.ravel())
# Plot showing the imbalances removed using SMOTE
sns.countplot(y_smote[1:])
# Second using BorderlineSMOTE for removing imbalance
bls = BorderlineSMOTE(random_state = 0)
x_bls, y_bls = bls.fit_sample(x_scaled_array, y_array.ravel())
# Plot showing the imbalances removed using BorderLineSMOTE
sns.countplot(y_bls[1:])
# Third using KMeansSMOTE for removing imbalance
svms = SVMSMOTE(random_state = 0)
x_svms, y_svms = svms.fit_sample(x_scaled_array, y_array.ravel())
# Plot showing the imbalances removed using BorderLineSMOTE
sns.countplot(y_svms[1:])
###############################################################################
# FEATURE SELECTION
# Recursive Feature Selection
# Converting the numpy arrays from balanced training dataset to dataframe
x_smote_df = pd.DataFrame(x_smote, columns = x_scaled.columns)
y_smote_df = pd.DataFrame(y_smote, columns = ['target'])
x_imp_smote = pd.DataFrame()
logreg = LogisticRegression(random_state=0, solver='saga')
rfe = RFE(logreg, 20)
rfe = rfe.fit(x_smote_df, y_smote_df.values.ravel())
for i in range(len(rfe.support_)):
    if rfe.support_[i] == True :
        print(x_scaled.columns[i])
        testdf = pd.DataFrame(x_smote_df[x_smote_df.columns[i]],
                              columns=[x_smote_df.columns[i]])
        x_imp_smote = pd.concat([x_imp_smote, testdf.reindex(testdf.index)],
                                 axis=1)
###############################################################################
# PRE-PROCESSING THE TESTING DATA AS THE TRAINING DATA
# Since the test data is already imputed only normalization will be carried out
# Separating independent and dependent variables
datacols = list(imputed_df_test)
y_test = imputed_df_test[datacols[1]]
x = imputed_df_test[datacols[2:]]
y_test = pd.DataFrame(y_test)
scaled_data = pd.DataFrame()
lcol = len(x.columns)
for i in range(lcol):
    data = x[x.columns[i]]
    data = data.values.astype(float)
    data = data.reshape(-1, 1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(data)
    testdf = pd.DataFrame(scaled_array, columns=[x.columns[i]])
    scaled_data = pd.concat([scaled_data, testdf.reindex(testdf.index)],
                            axis=1)
x_test_scaled = scaled_data
x_test_scaled.describe()
# Using the important features in the test data
x_test_imp = pd.DataFrame()
for i in range(len(rfe.support_)):
    if rfe.support_[i] == True :
        print(x_test_scaled.columns[i])
        testdf = pd.DataFrame(x_test_scaled[x_test_scaled.columns[i]],
                              columns=[x_test_scaled.columns[i]])
        x_test_imp = pd.concat([x_test_imp, testdf.reindex(testdf.index)],
                                 axis=1)
###############################################################################
# Using Logistic Regression to predict on test data with important features
logreg.fit(x_imp_smote, y_smote)
# Predicting on test data
y_test_pred = logreg.predict(x_test_imp)
logreg.score(x_test_imp, y_test)
confusion_matrix_logit = confusion_matrix(y_test, y_test_pred)
print(confusion_matrix_logit)
print(classification_report(y_test, y_test_pred))
###############################################################################
# Using Logistic Regression to predict on test data with all features
logreg.fit(x_smote, y_smote)
# Predicting on test data
y_test_pred = logreg.predict(x_test_scaled)
logreg.score(x_test_scaled, y_test)
confusion_matrix_logit = confusion_matrix(y_test, y_test_pred)
print(confusion_matrix_logit)
print(classification_report(y_test, y_test_pred))
# The results of the two predictions are almost the same hence the feature
# selection process is not improving our accuracy
###############################################################################
# XGBoost Modeling
data_dmatrix = xgb.DMatrix(data=x_smote,label=y_smote)
xgb_model = XGBClassifier(objective="binary:logistic", max_depth=170, 
                          n_estimators=50)
xgb_model.fit(x_smote_df, y_smote_df)
# Predicting on test data
y_test_pred = xgb_model.predict(x_test_scaled)
xgb_model.score(x_test_scaled, y_test)
confusion_matrix_xgb = confusion_matrix(y_test, y_test_pred)
print(confusion_matrix_xgb)
print(classification_report(y_test, y_test_pred))
accuracy = accuracy_score(y_test, y_test_pred)
print("Accuracy: %.2f%%" % (accuracy*100.0))
###############################################################################
# TENSORFLOW MODELING FOR CLASSIFICATION
tf_model = Sequential([
    Dense(1000, activation='relu'),
    Dropout(0.2),
    Dense(500, activation='relu'),
    Dropout(0.2),
    Dense(100, activation='relu'),
    Dense(10, activation='relu'),
    Dense(1, activation='sigmoid'),
])
tf_model.compile(optimizer='adam', loss='binary_crossentropy', 
                 metrics=['accuracy',])
# Training the model using data obtained by SMOTE technique
history_smote = tf_model.fit(x_smote, y_smote, epochs=15, batch_size=256)
tf_model.save("TF_Model_Conoco.h5")
x_test_scaled_array = x_test_scaled.to_numpy()
y_test_pred = tf_model.predict(x_test_scaled_array)
y_test_pred2 = np.array([int(i>=0.9999) for i in y_test_pred])
score = f1_score(y_test,y_test_pred2,average =None )
score
confusion_matrix_tf = confusion_matrix(y_test, y_test_pred2)
print(confusion_matrix_tf)
accuracy = accuracy_score(y_test, y_test_pred2)
print("Accuracy: %.2f%%" % (accuracy*100.0))
#
# Training the model using data obtained by BoundaryLineSMOTE technique
history_bls = tf_model.fit(x_bls, y_bls, epochs=15, batch_size=256)
tf_model.save("TF_Model_Conoco.h5")
x_test_scaled_array = x_test_scaled.to_numpy()
y_test_pred = tf_model.predict(x_test_scaled_array)
y_test_pred2 = np.array([int(i>=0.999999) for i in y_test_pred])
score = f1_score(y_test,y_test_pred2,average =None )
score
confusion_matrix_tf = confusion_matrix(y_test, y_test_pred2)
print(confusion_matrix_tf)
accuracy = accuracy_score(y_test, y_test_pred2)
print("Accuracy: %.2f%%" % (accuracy*100.0))
#
# Training the model using data obtained by SVMSMOTE technique
history_svms = tf_model.fit(x_svms, y_svms, epochs=15, batch_size=256)
tf_model.save("TF_Model_Conoco.h5")
x_test_scaled_array = x_test_scaled.to_numpy()
y_test_pred = tf_model.predict(x_test_scaled_array)
y_test_pred2 = np.array([int(i>=0.99999) for i in y_test_pred])
score = f1_score(y_test,y_test_pred2,average =None )
score
confusion_matrix_tf = confusion_matrix(y_test, y_test_pred2)
print(confusion_matrix_tf)
accuracy = accuracy_score(y_test, y_test_pred2)
print("Accuracy: %.2f%%" % (accuracy*100.0))


# ROC Curve
tf_roc_auc = roc_auc_score(y_test, y_test_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
plt.figure()
plt.plot(fpr, tpr, label='TF Keras Sequential (area = %0.2f)' % tf_roc_auc)
#plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('TF_ROC')
plt.show()

###############################################################################
# Loading new testing dataset from csv
df_newtest = pd.read_csv("equip_failures_test_set.csv")
# The data needs to be checked for any missing values. First lets gather how
# many missng values are present in the data.
missing_total = df_newtest.isnull().sum().sort_values(ascending=False)
missing_percent = ((df_newtest.isnull().sum()/df_newtest.isnull().count()).
                   sort_values(ascending=False))
missing_data = pd.concat([missing_total, missing_percent], axis=1,
                         keys=['Missing Total', 'Missing Percent'])
missing_data.head()
# We found no missing values in the data.
# Replacing string 'na' by numpy NaN values
nareplaced_df = df_newtest.replace('na', np.NaN)
# Lets gather how many NaN values are present in our dataset.
nan_total = nareplaced_df.isna().sum().sort_values(ascending=False)
nan_percent = ((nareplaced_df.isna().sum()/nareplaced_df.isna().count()).
               sort_values(ascending=False))
nan_data = pd.concat([nan_total, nan_percent], axis=1,
                     keys=['NaN Total', 'NaN Percent'])
nan_data.head(50)
# There are a lot of nan values in some of the variables of the dataset.
# Convert the data type of the variables to float
float_df = nareplaced_df.astype(float)
float_df.describe()
# First scheme of imputation
# Impute the remaining variables with their mean
# Using fillna() function creates a dataframe
imputed_df = float_df.fillna(float_df.mean())
datacols_test = list(imputed_df)
x_newtest = imputed_df[datacols_test[1:]]
# Normalizing the new test data
scaled_data = pd.DataFrame()
lcol = len(x_newtest.columns)
for i in range(lcol):
    data = x_newtest[x_newtest.columns[i]]
    data = data.values.astype(float)
    data = data.reshape(-1,1)
    min_max_scaler = preprocessing.MinMaxScaler()
    scaled_array = min_max_scaler.fit_transform(data)
    testdf = pd.DataFrame(scaled_array,columns=[x_newtest.columns[i]])
    scaled_data=pd.concat([scaled_data, testdf.reindex(testdf.index)], axis=1)
x_scaled_data_newtest = scaled_data
# Predicting on new test data
x_scaled_newtest_array = x_scaled_data_newtest.to_numpy()
y_newtest_pred = tf_model.predict(x_scaled_newtest_array)
y_newtest_pred2 = np.array([int(i>=0.99999) for i in y_newtest_pred])
pred_df = pd.DataFrame(y_newtest_pred2,columns=['target'])
pred_df.reset_index(level=0, inplace=True)
pred_df.columns =['id','target']
pred_df.to_csv('myPrediction_onTestData_UsingTF.csv',index=False)
