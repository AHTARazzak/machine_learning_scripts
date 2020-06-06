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

#This script determines the energetic barrier of a ligand transition depending on kinetic and thermodynamic features of the protein and system.
#Intended to be used in CHARMM pipeline but can be more generally applied to other molecular dynamic pipelines.

#Load in data with all features.
df=pd.read_csv('0100_mig_features_phaseproper.csv',delimiter=",")
df.head()

#Check the PMF.
df['PMF'].nunique()

#Want to predict PMF.
y = df[['PMF']]
#Features to determine PMF.
x = df[['cav1','cav2','cav3','cav4','rmsd','energyself','energywat']]

#Scale the variables.
x['cav1']=(df['cav1']-df['cav1'].mean())/df['cav1'].std().round(4)
x['cav2']=(df['cav2']-df['cav2'].mean())/df['cav2'].std().round(4)
x['cav3']=(df['cav3']-df['cav3'].mean())/df['cav3'].std().round(4)
x['cav4']=(df['cav4']-df['cav4'].mean())/df['cav4'].std().round(4)
x['rmsd']=((df['rmsd']-df['rmsd'].mean())/df['rmsd'].std()).round(3)
x['energyself']=(df['energyself']-df['energyself'].mean())/df['energyself'].std().round(4)
x['energywat']=(df['energywat']-df['energywat'].mean())/df['energywat'].std().round(4)

#Encode variables.
enc=preprocessing.LabelEncoder()
y['PMF'] = enc.fit_transform(y['PMF'])
x['cav1'] = enc.fit_transform(x['cav1'])
x['cav2'] = enc.fit_transform(x['cav2'])
x['cav3'] = enc.fit_transform(x['cav3'])
x['cav4'] = enc.fit_transform(x['cav4'])
x['rmsd'] = enc.fit_transform(x['rmsd'])
x['energyself'] = enc.fit_transform(x['energyself'])
x['energywat'] = enc.fit_transform(x['energywat'])

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

#Train and fit KNN model (k=13).
KNNmod=KNeighborsClassifier(n_neighbors=13)
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

#Determine best gamma value
gamma_list=[]
for i in range(1,10):
  classifier=SVC(kernel='rbf',gamma=(float(i)/10))
  classifier.fit(X_train,y_train)
  gamma_list.append(classifier.score(X_test,y_test))
print(gamma_list)

#Pick gamma where curve flattens.
plt.plot(range(1,10),gamma_list)
plt.ylabel('Validation acc')
plt.title('nice')
plt.show()


#Train and fit Support Vector Machine model (gamma=0.1).
SVCmod=SVC(kernel='linear',gamma=0.1)
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

#Train and fit Decision Tree model.
DTmod=DecisionTreeClassifier(criterion='gini', random_state = 77,max_depth=10)
DTmod.fit(X_train,y_train)
#Score DT fit.
DTmod.score(X_train, y_train)

#Predict test and score of Decision Tree model.
DTpred = DTmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,DTpred)*100)
print('F1 score:', f1_score(y_test, DTpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, DTpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, DTpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, DTpred))
print('Classification report:', classification_report(y_test, DTpred))

#Plot feature weighting from Decision Tree.
featurelist=list(df.columns[1:])
plt.bar(range(len(DTmod.feature_importances_)), DTmod.feature_importances_)
plt.xlabel("feature",fontsize=14)
plt.ylabel("importance",fontsize=14)
plt.title("feature importance", fontsize=18)
plt.xticks(range(len(DTmod.feature_importances_)), featurelist,rotation='vertical')
plt.show()
print(featurelist)

#Train and fit Random Forest model.
RFmod=RandomForestClassifier(n_estimators=100,random_state=77)
RFmod.fit(X_train,y_train)
#Score RF fit.
RFmod.score(X_train, y_train)

#Predict test and score of Random Forest model.
RFpred = RFmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,RFpred)*100)
print('F1 score:', f1_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Recall score:', recall_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Precision score:', precision_score(y_test, RFpred,pos_label='positive',average='macro')*100)
print('Confusion matrix:', confusion_matrix(y_test, RFpred))
print('Classification report:', classification_report(y_test, RFpred))

#Plot feature weighting from Random Forest.
featurelist=list(df.columns[1:])
plt.bar(range(len(RFmod.feature_importances_)), RFmod.feature_importances_)
plt.xlabel("Feature", fontsize=14, family='Arial')
plt.ylabel("Importance", fontsize=14, family='Arial')
plt.title("Feature importance", fontsize=18, family='Arial')
plt.xticks(range(len(RFmod.feature_importances_)), featurelist,rotation='vertical')
plt.xticks(np.arange(7), ('Xe1', 'Xe2', 'Xe3', 'Xe4', 'RMSD','Energy','WaterEnergy'))
plt.show()
print(featurelist)

#K-fold comparison between models
kfold = model_selection.KFold(n_splits=10, random_state=77)
cv_LoR = model_selection.cross_val_score(LogisticRegression(max_iter=10000), X_train, y_train, cv=kfold, scoring='accuracy')
cv_DT = model_selection.cross_val_score(DecisionTreeClassifier(criterion='gini', random_state = 77,max_depth=10), X_train, y_train, cv=kfold, scoring='accuracy')
cv_RF = model_selection.cross_val_score(RandomForestClassifier(n_estimators=100), X_train, y_train, cv=kfold, scoring='accuracy')
#cv_SVC = model_selection.cross_val_score(SVC(kernel='rbf',gamma=0.1), X_train, y_train, cv=kfold, scoring='accuracy')
cv_KNN = model_selection.cross_val_score(KNeighborsClassifier(n_neighbors=7), X_train, y_train, cv=kfold, scoring='accuracy')
results=[cv_LoR,cv_DT, cv_RF, cv_KNN]
names=["LoR","DT","RF","KNN"]
fig = plt.figure()
fig.suptitle('Algorithm Comparison',fontsize=18)
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()
