import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix
from sklearn import tree
from sklearn import svm
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
import matplotlib
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import r2_score
import statsmodels.api as sm
from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso


#This script determines the position of a ligand depending on various features of a molecular protein system.
#Intended to be used in CHARMM pipeline but can be more generally applied to other molecular dynamic pipelines.
#Several different data frames with different data.

#Load in data with all features.
df=pd.read_csv('xe2xe1_0100_track_cavall4.csv',delimiter=" ")
df.head(100)

#Want to predict location of atom.
y = df[['location']]
#Features to determine atom location.
x = df[['cav1','cav2','cav3','cav4']]

#Split data into train and test.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

#Train and fit Linear Regression model.
lm = LinearRegression()
lm.fit(X_train, y_train)
lm.score(X_train, y_train)

#Score Logistic regression fit.
lm.score(X_test, y_test)

print("The linear model is: Y = {:.5} + {:.5}*cav1 + {:.5}*cav2 + {:.5}*cav3+ {:.5}*cav4".format(lm.intercept_[0], lm.coef_[0][0], lm.coef_[0][1], lm.coef_[0][2], lm.coef_[0][3]))

#
X = np.column_stack((df['cav1'], df['cav2'], df['cav3'],df['cav4']))
y = df['location']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

#Use Lasso model to score fit.
reg = LassoCV()
reg.fit(x, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(x,y))
coef = pd.Series(reg.coef_, index = x.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()
import matplotlib
matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)
imp_coef.plot(kind = "barh")
plt.title("Feature importance using Lasso Model")

#Load in data with all features.
df=pd.read_csv('xe2xe1_0100_track_cavall4.csv',delimiter=" ")

#Want to predict location of atom.
y = df[['location']]
#Features to determine atom location.
x = df[['cav1','cav2','cav3','cav4','rmsd','engself','engwat']]

#Split data into train and test.
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

#Train and fit Logistic Regression model.
LoRmod=LogisticRegression(max_iter=10000)
LoRmod.fit(X_train,y_train)
#Score LR training
LoRmod.score(X_train, y_train)

#Predict test and score of Logistic regression model.
LoRpred= LoRmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,LoRpred)*100)
print('F1 score:', f1_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, LoRpred))
print('Classification report:', classification_report(y_test, LoRpred))

#Determine best k-value.
k_list=[]
for i in range(1,20):
  classifier=KNeighborsClassifier(n_neighbors=i)
  classifier.fit(X_train,y_train)
  k_list.append(classifier.score(X_test,y_test))

#Pick k-value where curve flattens.
plt.plot(range(1,20),k_list)
plt.ylabel('Validation acc')
plt.title('nice')
plt.show()

#Train and fit KNN model (k=15).
KNNmod=KNeighborsClassifier(n_neighbors=15)
KNNmod.fit(X_train,y_train)
#Score KNN fit.
KNNmod.score(X_train, y_train)

#Predict test and score of K-nearest neighbour model.
KNNpred = KNNmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,KNNpred)*100)
print('F1 score:', f1_score(y_test, KNNpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, KNNpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, KNNpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, KNNpred))
print('Classification report:', classification_report(y_test, KNNpred))

#Train and fit Support Vector Machine model (gamma=0.1).
SVCmod=SVC(kernel='rbf',gamma=0.1)
SVCmod.fit(X_train,y_train)
#Score SVC fit.
SVCmod.score(X_train, y_train)

#Predict test and score of SVC model.
SVCpred = SVCmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,SVCpred)*100)
print('F1 score:', f1_score(y_test, SVCpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, SVCpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, SVCpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, SVCpred))
print('Classification report:', classification_report(y_test, SVCpred))

#Train and fit Random Forest model.
RFmod=RandomForestClassifier(n_estimators=100,random_state=77)
RFmod.fit(X_train,y_train)
#Score RF fit.
RFmod.score(X_train, y_train)

#Predict test and score of RF model.
RFpred = RFmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,RFpred)*100)
print('F1 score:', f1_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, RFpred))
print('Classification report:', classification_report(y_test, RFpred))

#Load data & do it all over again.
df=pd.read_csv('xe2xe1_0100_track_cavall4.csv',delimiter=" ")
y = df[['location']]
x = df[['cav1','cav2','cav3','cav4','rmsd','engself','engwat']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

lm = LinearRegression()
lm.fit(X_train, y_train)
lm.score(X_train, y_train)
lm.score(X_test, y_test)

print("The linear model is: Y = {:.5} + {:.5}*cav1 + {:.5}*cav2 + {:.5}*cav3+ {:.5}*cav4".format(lm.intercept_[0], lm.coef_[0][0], lm.coef_[0][1], lm.coef_[0][2], lm.coef_[0][3]))


X = np.column_stack((df['cav1'], df['cav2'], df['cav3'],df['cav4']))
y = df['location']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())


X = np.column_stack((df['cav1'], df['cav2'], df['cav3'],df['cav4']))
y = df['location']
X2 = sm.add_constant(X)
est = sm.OLS(y, X2)
est2 = est.fit()
print(est2.summary())

df=pd.read_csv('xe2xe1_0100_track_cavall4.csv',delimiter=" ")

y = df[['location']]
x = df[['cav1','cav2','cav3','cav4']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

LoRmod=LogisticRegression(max_iter=10000)
LoRmod.fit(X_train,y_train)
#Score LR training
LoRmod.score(X_train, y_train)

LoRpred= LoRmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,LoRpred)*100)
print('F1 score:', f1_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, LoRpred))
print('Classification report:', classification_report(y_test, LoRpred))

df=pd.read_csv('xe2xe1_0100_track_cavall4.csv',delimiter=" ")

y = df[['location']]
x = df[['cav1','cav2','cav3','cav4','rmsd','engself','engwat']]

cav1mean=x['cav1'].mean()

x['cav1']=abs(df['cav1']-df['cav1'].mean())/df['cav1'].std().round(4)
x['cav2']=abs(df['cav2']-df['cav2'].mean())/df['cav2'].std().round(4)
x['cav3']=abs(df['cav3']-df['cav3'].mean())/df['cav3'].std().round(4)
x['cav4']=abs(df['cav4']-df['cav4'].mean())/df['cav4'].std().round(4)
x['cav4']=abs(df['rmsd']-df['rmsd'].mean())/df['rmsd'].std()

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

LoRmod=LogisticRegression(max_iter=10000)
LoRmod.fit(X_train,y_train)
LoRmod.score(X_train, y_train)
LoRpred= LoRmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,LoRpred)*100)
print('F1 score:', f1_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, LoRpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, LoRpred))
print('Classification report:', classification_report(y_test, LoRpred))

k_list=[]
for i in range(1,20):
  classifier=KNeighborsClassifier(n_neighbors=i)
  classifier.fit(X_train,y_train)
  k_list.append(classifier.score(X_test,y_test))
plt.plot(range(1,20),k_list)
plt.ylabel('Validation acc')
plt.title('nice')
plt.show()
KNNmod=KNeighborsClassifier(n_neighbors=15)
KNNmod.fit(X_train,y_train)
KNNmod.score(X_train, y_train)
KNNpred = KNNmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,KNNpred)*100)
print('F1 score:', f1_score(y_test, KNNpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, KNNpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, KNNpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, KNNpred))
print('Classification report:', classification_report(y_test, KNNpred))

gamma_list=[]
for i in range(1,10):
  classifier=SVC(kernel='rbf',gamma=(float(i)/10))
  classifier.fit(X_train,y_train)
  gamma_list.append(classifier.score(X_test,y_test))
print(gamma_list)
plt.plot(range(1,10),gamma_list)
plt.ylabel('Validation acc')
plt.title('nice')
plt.show()
SVCmod=SVC(kernel='linear',gamma=0.1)
SVCmod.fit(X_train,y_train)
#Score DT fit.
SVCmod.score(X_train, y_train)
SVCpred = SVCmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,SVCpred)*100)
print('F1 score:', f1_score(y_test, SVCpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, SVCpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, SVCpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, SVCpred))
print('Classification report:', classification_report(y_test, SVCpred))


DTmod=DecisionTreeClassifier(criterion='gini', random_state = 77,max_depth=10)
DTmod.fit(X_train,y_train)
#Score DT fit.
DTmod.score(X_train, y_train)
DTpred = DTmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,DTpred)*100)
print('F1 score:', f1_score(y_test, DTpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, DTpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, DTpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, DTpred))
print('Classification report:', classification_report(y_test, DTpred))
featurelist=list(df.columns[1:])
plt.bar(range(len(DTmod.feature_importances_)), DTmod.feature_importances_)
plt.xlabel("feature",fontsize=14)
plt.ylabel("importance",fontsize=14)
plt.title("feature importance", fontsize=18)
plt.xticks(range(len(DTmod.feature_importances_)), featurelist,rotation='vertical')
plt.show()
print(featurelist)

RFmod=RandomForestClassifier(n_estimators=100,random_state=77)
RFmod.fit(X_train,y_train)
#Score RF fit.
RFmod.score(X_train, y_train)
RFpred = RFmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,RFpred)*100)
print('F1 score:', f1_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, RFpred))
print('Classification report:', classification_report(y_test, RFpred))
featurelist=list(df.columns[1:])
plt.bar(range(len(RFmod.feature_importances_)), RFmod.feature_importances_)
plt.xlabel("feature", fontsize=14)
plt.ylabel("importance", fontsize=14)
plt.title("feature importance", fontsize=18)
plt.xticks(range(len(RFmod.feature_importances_)), featurelist,rotation='vertical')
plt.show()
print(featurelist)

kfold = model_selection.KFold(n_splits=10, random_state=77)
cv_LoR = model_selection.cross_val_score(LogisticRegression(max_iter=10000), X_train, y_train, cv=kfold, scoring='accuracy')
cv_DT = model_selection.cross_val_score(DecisionTreeClassifier(criterion='gini', random_state = 77,max_depth=10), X_train, y_train, cv=kfold, scoring='accuracy')
cv_RF = model_selection.cross_val_score(RandomForestClassifier(n_estimators=100), X_train, y_train, cv=kfold, scoring='accuracy')
#cv_SVC = model_selection.cross_val_score(SVC(kernel='rbf',gamma=0.1), X_train, y_train, cv=kfold, scoring='accuracy')
cv_KNN = model_selection.cross_val_score(KNeighborsClassifier(n_neighbors=7), X_train, y_train, cv=kfold, scoring='accuracy')
results=[cv_LoR,cv_DT, cv_RF,cv_KNN]
names=["LoR","DT","RF","KNN"]
fig = plt.figure()
fig.suptitle('Algorithm Comparison',fontsize=18)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Again with different predictor preprocessing.
df['location']=df['location'].apply(lambda x: 2 if x==2 else 3)
y = df[['location']]
x = df[['cav1','cav2','cav3','cav4','rmsd','engself','engwat']]

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=77)

RFmod=RandomForestClassifier(n_estimators=100,random_state=77)
RFmod.fit(X_train,y_train)
#Score RF fit.
RFmod.score(X_train, y_train)
RFpred = RFmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,RFpred)*100)
print('F1 score:', f1_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, RFpred))
print('Classification report:', classification_report(y_test, RFpred))
featurelist=list(df.columns[1:])
plt.bar(range(len(RFmod.feature_importances_)), RFmod.feature_importances_)
plt.xlabel("feature", fontsize=14)
plt.ylabel("importance", fontsize=14)
plt.title("feature importance", fontsize=18)
plt.xticks(range(len(RFmod.feature_importances_)), featurelist,rotation='vertical')
plt.show()
print(featurelist)
