import pickle
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from preprocessing import *
import matplotlib.pyplot as plt
from sklearn.experimental import  enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import  RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import svm
data = pd.read_csv('loan_data.csv')
####################################################################################
print(data.isna().sum())
# filling nulls
for i in [data]:
    i["Gender"] = i["Gender"].fillna(data.Gender.dropna().mode()[0])
    i["Married"] = i["Married"].fillna(data.Married.dropna().mode()[0])
    i["Dependents"] = i["Dependents"].fillna(data.Dependents.dropna().mode()[0])
    i["Self_Employed"] = i["Self_Employed"].fillna(data.Self_Employed.dropna().mode()[0])
    i["Credit_History"] = i["Credit_History"].fillna(data.Dependents.dropna().mode()[0])
    i["LoanAmount"] = i["LoanAmount"].fillna(i["LoanAmount"].mean())
    i["Loan_Amount_Term"] = i["Loan_Amount_Term"].fillna(i["Loan_Amount_Term"].mode()[0])
# impute to fill nulls in LoanAmount and Loan_Amount_Term
'''x1=data.loc[:,['LoanAmount','Loan_Amount_Term']]
imp=IterativeImputer(RandomForestRegressor(),max_iter=10,random_state=0)
x1=pd.DataFrame(imp.fit_transform(x1),columns=x1.columns)
data['LoanAmount']=x1['LoanAmount']
data['Loan_Amount_Term']=x1['Loan_Amount_Term']'''
# print(data['Loan_Amount_Term'].head(22))
# print(data.isnull().sum())
###########################################################################
x = data.iloc[:, 0:12]  # features
y = data['Loan_Status']  # target
# remove duplicates
x.drop_duplicates(subset="Loan_ID", keep=False, inplace=True)
# drop loan_id
x.drop("Loan_ID", axis=1, inplace=True)
###############################################################################
# converting categorical data
x['Credit_History'] = x['Credit_History'].astype(int)
# label encoding
cols = ('Gender', 'Married', 'Dependents', 'Self_Employed', 'Property_Area', 'Education')
x = Feature_Encoder(x, cols)
# mapping target to 0 and 1
y = y.map({"N": 0, "Y": 1}).astype(int)
##################################################################################
# visualization
# fig,ax=(plt.subplots(2,4,figsize=(16,10)))
# sns.countplot(y,data=y,ax=ax[0][0])
# sns.countplot('Gender',data=x,ax=ax[0][1])
# sns.countplot('Married',data=x,ax=ax[0][2])
# sns.countplot('Self_Employed',data=x,ax=ax[1][0])
# sns.countplot('Property_Area',data=x,ax=ax[1][1])
# sns.countplot('Credit_History',data=x,ax=ax[1][2])
# sns.countplot('Dependents',data=x,ax=ax[1][3])
# sns.countplot('Education',data=x,ax=ax[0][3])
# plt.show()
########################################################################
# featureScaling
stand = MinMaxScaler()
col = x.columns.tolist()
x[col] = stand.fit_transform(x[col])
########################################################################
# splitting data to 20% test and 80 % trian
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.20,shuffle=True,random_state=40)
###########################################################################
# SVM model
C = 0.1  # SVM regularization parameter
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
accuracy = svc.score(X_test,y_test)
print('SVC test accuracy '+ str(accuracy))
y_pred = svc.predict(X_test)
print("svm accuracy ",metrics.accuracy_score(y_test,y_pred))
print("svm precision ",metrics.precision_score(y_test,y_pred))
print("svm recall ",metrics.recall_score(y_test,y_pred))
###############################################################################
# logistic regression
from sklearn.model_selection import cross_val_score
log = LogisticRegression()
cross_val_score(log, X_train, y_train, scoring=make_scorer(accuracy_score), cv=3)
pred = log.fit(X_train,y_train).predict(X_test)
print("logistic regression accuracy ",accuracy_score(y_test,pred))
print("precision : ",metrics.precision_score(y_test,pred))
print("Recall : ",metrics.recall_score(y_test,pred))
######################################################################################
#decision tree
from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)
predictions = model.predict(X_test)
#print(classification_report(y_test, predictions))
print("desicion tree accuracy:",accuracy_score(y_test,predictions))
#######################################################################################3
# random forrest
classifier = RandomForestClassifier(n_estimators = 50)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)
print("random forrest accuracy:",accuracy_score(y_test,y_pred))
