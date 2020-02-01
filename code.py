# --------------
# Importing Necessary libraries
import warnings
warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
plt.rcParams['figure.figsize'] = (10, 8)
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Imputer

# Load the train data stored in path variable
train = pd.read_csv(path)
# Load the test data stored in path1 variable
test = pd.read_csv(path1)
# necessary to remove rows with incorrect labels in test dataset
classes_train = [' <=50K', ' >50K']
classes_test = [' <=50K.', ' >50K.']
test = test[test['Target'].isin(classes_test)]
train = train[train['Target'].isin(classes_train)]
#print(test['Target'])
# encode target variable as integer
train.loc[train['Target'] == ' <=50K', 'Target'] = 0
train.loc[train['Target'] == ' >50K', 'Target'] = 1
test.loc[test['Target'] == ' <=50K.', 'Target'] = 0
test.loc[test['Target'] == ' >50K.', 'Target'] = 1
# Plot the distribution of each feature
#fig = plt.figure(figsize=(20,30))
#rows = 5
#cols = 3
#for i, column in enumerate(train.columns) :
#    ax = fig.add_subplot(rows, cols, i+1)
#    ax.set_title(column)
#    if train.dtypes[column] == "object" :
#        train[column].value_counts().plot(kind='bar')
#    else :
#        train[column].hist()
# convert the data type of Age column in the test data to int type
test['Age'] = test['Age'].astype(int)
# cast all float features to int type to keep types consistent between our train and test data
for column in test.columns :
    if test.dtypes[column] == "float64" :
        test[column] = test[column].astype(int)
# choose categorical and continuous features from data and print them
cat = train.select_dtypes(include=['object']).columns
num = train.select_dtypes(exclude=['object']).columns
# fill missing data for catgorical columns
for col in cat :
        train[col].fillna(train[col].mode()[0], inplace=True)
        test[col].fillna(test[col].mode()[0], inplace=True)
#mode_imputer = Imputer(strategy='most_frequent')
#mode_imputer.fit_transform(train_cat)
#train_cat = mode_imputer.transform(train)
# fill missing data for numerical columns   
for col in num :
        train[col].fillna(train[col].median(), inplace=True)
        test[col].fillna(test[col].median(), inplace=True)
#median_imputer = Imputer(strategy='median')
#median_imputer.fit_transform(train_num)
#train_num = median_imputer.transform(train_num)
# Dummy code Categoricol features
le = LabelEncoder()
for col in cat :
        train[col] = le.fit_transform(train[col])
        test[col] = le.transform(test[col])
train = pd.concat([train[num], pd.get_dummies(data=train, columns=cat)], axis=1) 
test = pd.concat([test[num], pd.get_dummies(data=test, columns=cat)], axis=1)
# Check for Column which is not present in test data
col_nottest = train.columns[~train.columns.isin(test.columns)]
# New Zero valued feature in test data for Holand
test['Country_14'] = 0
# Split train and test data into X_train ,y_train,X_test and y_test data
X_train = train.drop(['Target'], axis=1)
y_train = train['Target']
X_test = test.drop(['Target'], axis=1)
y_test = test['Target']
# train a decision tree model then predict our test data and compute the accuracy
dt = DecisionTreeClassifier(max_depth=3, random_state=17)
dt.fit(X_train, y_train)
tree_score = dt.score(X_test, y_test)
# Decision tree with parameter tuning
tree_params = {'max_depth': range(2,11)}
dt1 = DecisionTreeClassifier(random_state=17)
p_tree = GridSearchCV(dt1, tree_params, cv=5)
p_tree.fit(X_train, y_train)
p_score = p_tree.score(X_test, y_test)
# Print out optimal maximum depth(i.e. best_params_ attribute of GridSearchCV) and best_score_
print(p_tree.best_params_)
print(p_tree.best_score_)
#train a decision tree model with best parameter then predict our test data and compute the accuracy
dt2 = DecisionTreeClassifier(max_depth=9, random_state=17)
dt2.fit(X_train, y_train)
test_acc = dt2.score(X_test, y_test)
print(test_acc)


