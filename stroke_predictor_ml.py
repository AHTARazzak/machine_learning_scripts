import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report, precision_score, recall_score, confusion_matrix
from sklearn import tree
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler,SMOTE, ADASYN
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

#This script used clinical data to predict stroke outcome.

#Import data & cursory inspection of data
strokedata_train=pd.read_csv('train_2v.csv')
strokedata_test=pd.read_csv('test_2v.csv')
strokedata_train.head(5)

#Inspecting and cleaning data.
#drop id feature (not informative of stroke)
strokedf_train=strokedata_train.drop('id',axis=1)
strokedf_train.describe()
strokedf_test=strokedata_test.drop('id',axis=1)
strokedf_test.describe()

#Inspect catagories within data to see nature of classifications and why count differences between features.
print(strokedf_train['gender'].unique())
print(strokedf_train['hypertension'].unique())
print(strokedf_train['heart_disease'].unique())
print(strokedf_train['ever_married'].unique())
print(strokedf_train['work_type'].unique())
print(strokedf_train['Residence_type'].unique())
print(strokedf_train['smoking_status'].unique())
print(strokedf_train['stroke'].unique())
print(strokedf_train['age'].min())

#Notced null values, searching for others.
#identify number of null values in dataframe.
strokedf_train.isnull().sum()

#Deciding what to do with null features.
#Identify how many of the features with null values are not null.
print("BMI # avaliable (%):"+str(len(strokedf_train)-1462) + '('+str((len(strokedf_train)-1462)/len(strokedf_train))+')')
print("smoking # avaliable (%):"+ str(len(strokedf_train)-13292) +'('+str((len(strokedf_train)-13292)/len(strokedf_train))+')')


#Considering removing entries with NULL bmi value (3% of data) or retaining.
sns.boxplot(x='stroke', y='bmi', data=strokedf_train)

#Inspect distribution of gender (Noticed "Other" classification).
sns.countplot(strokedf_train['gender'])

#Want to see "Other" contribution to data.
strokedf_train['gender'].str.count("Other").sum()

#Decide to remove "Other" from data on grounds that:
#1) Paucity of data (11 values) thereby difficult to reliably predict relation to stroke using ML.
#2) "Other" as a gender category in itself is ambiguous in that subject could be male to female, or female to male, or non-gender identifying. Thereby the potentialf or at least three different categories within that 11.
#3) Difficulty finding scientific literature which inspects causation between not identifying as a male or female and stroke prevalance.
strokedf_train = strokedf_train[strokedf_train['gender'] != "Other"]
strokedf_test = strokedf_test[strokedf_test['gender'] != "Other"]
sns.countplot(strokedf_train['gender'])

#Inspecting distribution of other features with regards to stroke .
sns.catplot(x='hypertension',kind='count',data=strokedf_train, hue='stroke', legend=False)
plt.legend(loc='upper right', labels=['No-stroke', 'Stroke'])
sns.catplot(x='heart_disease',kind='count',hue='stroke', legend=False,data=strokedf_train)
plt.legend(loc='upper right', labels=['No-stroke', 'Stroke'])
sns.catplot(x='ever_married',kind='count',hue='stroke', legend=False,data=strokedf_train)
plt.legend(loc='upper right', labels=['No-stroke', 'Stroke'])
sns.catplot(x='work_type',kind='count',hue='stroke', legend=False,data=strokedf_train)
plt.legend(loc='upper right', labels=['No-stroke', 'Stroke'])
sns.catplot(x='Residence_type',kind='count',hue='stroke', legend=False,data=strokedf_train)
plt.legend(loc='upper right', labels=['No-stroke', 'Stroke'])
sns.catplot(x='smoking_status',kind='count',hue='stroke', legend=False,data=strokedf_train)
plt.legend(loc='upper right', labels=['No-stroke', 'Stroke'])
sns.distplot(strokedf_train.loc[strokedf_train['stroke']==0]['age'], label='No stroke',norm_hist=True, bins=20)
sns.distplot(strokedf_train.loc[strokedf_train['stroke']==1]['age'], label='Stroke',norm_hist=True, bins=20)
plt.legend()

#Decesion to remove samples with age less than 16 because:
#1) Many of the features in the data are strictly dependent on age being over 16 (marriage, smoking, employment).
#2) Research investigating causation between age and stroke typically omits ages under 15.
#3) Science has demonstrated that genetic predisposition (e.g. sickle cell anemia), environment (e.g. carbon monoxide levels), and conginental conditions are related to stroke in children.
#https://www.chp.edu/our-services/brain/neurology/stroke-program/causes-and-symptoms.
strokedf_train = strokedf_train[strokedf_train['age'] > 16]
strokedf_test = strokedf_test[strokedf_test['age'] > 16]
sns.distplot(strokedf_train.loc[strokedf_train['stroke']==0]['age'], label='No stroke',norm_hist=True, bins=20)
sns.distplot(strokedf_train.loc[strokedf_train['stroke']==1]['age'], label='Stroke',norm_hist=True, bins=20)
plt.legend()

#Non-uniform distribution of ages.
#Considering grouping data by age.
strokedf_train['age'].nunique()

sns.distplot(strokedf_train.loc[strokedf_train['stroke']==0]['avg_glucose_level'], label='No stroke',norm_hist=True, bins=20)
sns.distplot(strokedf_train.loc[strokedf_train['stroke']==1]['avg_glucose_level'], label='Stroke',norm_hist=True, bins=20)
plt.legend()

#Data shows bimodal distribution for both stroke cases.
#Considering grouping data by blood glucose risk groups.
strokedf_train['avg_glucose_level'].nunique()

sns.distplot(strokedf_train.loc[strokedf_train['stroke']==0]['bmi'], label='No stroke', norm_hist=True, bins=20)
sns.distplot(strokedf_train.loc[strokedf_train['stroke']==1]['bmi'], label='Stroke', norm_hist=True, bins=20)
plt.legend()

#Data slightly right skewed.
#Considering organising BMI by groups.
strokedf_train['bmi'].nunique()

#Chose to retain BMI and replaced null values with median due to number of outliers.
#Considered predicting BMI however do not possess metrics to explicitly calculate.
#Consider possibility of predicting BMI using remaining data albeit may converge on predisposed bias in data.
strokedf_train['bmi'].fillna(strokedf_train['bmi'].median(), inplace= True)
strokedf_test['bmi'].fillna(strokedf_train['bmi'].median(), inplace= True)
#Considering removing data with smoking status or replacing with unknown.
#However null smoking status constitutes 30% of data.
#Chose to retain and assign "unknown".
strokedf_train['smoking_status'].fillna(value='unknown',inplace=True)
strokedf_test['smoking_status'].fillna(value='unknown',inplace=True)

#Checking if null values remain
strokedf_train.isnull().sum()

#Checking values across dataframe
strokedf_train.info()

#Paucity of data relating to stroke occurances, caution of potential overfitting to non-stroke due to imbalance in data.
sns.countplot(strokedf_train['stroke'])

#For building a decesion tree features need to be numerical
enc=preprocessing.LabelEncoder()

strokedf_train['gender'] = enc.fit_transform(strokedf_train['gender'])
strokedf_train['ever_married'] = enc.fit_transform(strokedf_train['ever_married'])
strokedf_train['work_type'] = enc.fit_transform(strokedf_train['work_type'])
strokedf_train['Residence_type'] = enc.fit_transform(strokedf_train['Residence_type'])
strokedf_train['smoking_status'] = enc.fit_transform(strokedf_train['smoking_status'])
strokedf_test['gender'] = enc.fit_transform(strokedf_test['gender'])
strokedf_test['ever_married'] = enc.fit_transform(strokedf_test['ever_married'])
strokedf_test['work_type'] = enc.fit_transform(strokedf_test['work_type'])
strokedf_test['Residence_type'] = enc.fit_transform(strokedf_test['Residence_type'])
strokedf_test['smoking_status'] = enc.fit_transform(strokedf_test['smoking_status'])

#Clear imbalance in stroke data as mentioned earlier, going to address now before ML to mitigate overfitting.
print(strokedf_train['stroke'].value_counts())
#ros=RandomOverSampler(random_state=77)
smote = SMOTE(random_state=77)
#Use ROSE method
X_resamp, y_resamp=smote.fit_resample(strokedf_train.loc[:,strokedf_train.columns!='stroke'],strokedf_train['stroke'])
print(format(X_resamp.shape))
print(format(y_resamp.shape))
y_resamp.value_counts()

#Split and train data
X_train, X_test, y_train, y_test = train_test_split(X_resamp, y_resamp, test_size=0.2, random_state=77)

#Train and fit Logistic Regression model.
LoRmod=LogisticRegression(max_iter=10000)
LoRmod.fit(X_train,y_train)
#Score LR training
LoRmod.score(X_train, y_train)

#Predict from manufactured test and score
LoRpred= LoRmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,LoRpred)*100)
print('F1 score:', f1_score(y_test, LoRpred)*100)
print('Recall score:', recall_score(y_test, LoRpred)*100)
print('Precision score:', precision_score(y_test, LoRpred)*100)
print('Confusion matrix:', confusion_matrix(y_test, LoRpred))
print('Classification report:', classification_report(y_test, LoRpred))

#Predict stroke in case study using LoR.
LoRprestest = LoRmod.predict(strokedf_test)
LoRpred = pd.DataFrame(LoRprestest,columns=['prediction'])
print(len(strokedf_test))
LoRpred['prediction'].value_counts()

#Train and fit Decesion tree model.
DTmod=DecisionTreeClassifier(criterion='gini', random_state = 77,max_depth=10)
DTmod.fit(X_train,y_train)
#Score DT fit.
DTmod.score(X_train, y_train)

#Predict from manufactured test and score
DTpred = DTmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,DTpred)*100)
print('F1 score:', f1_score(y_test, DTpred)*100)
print('Recall score:', recall_score(y_test, DTpred)*100)
print('Precision score:', precision_score(y_test, DTpred)*100)
print('Confusion matrix:', confusion_matrix(y_test, DTpred))
print('Classification report:', classification_report(y_test, DTpred))

#Determine how DT weighted features.
featurelist=list(strokedf_train.columns[:])
plt.bar(range(len(DTmod.feature_importances_)), DTmod.feature_importances_)
plt.xlabel("feature")
plt.ylabel("importance")
plt.title("feature importance")
plt.xticks(range(len(DTmod.feature_importances_)), featurelist,rotation='vertical')
plt.show()
print(featurelist)

#Predict stroke in case study using DT.
DTprestest = DTmod.predict(strokedf_test)
DTpredf = pd.DataFrame(DTprestest,columns=['prediction'])
print(len(strokedf_test))
DTpredf['prediction'].value_counts()

#Train and fit Random Forest model.
RFmod=RandomForestClassifier(n_estimators=100,random_state=77)
RFmod.fit(X_train,y_train)
#Score RF fit.
RFmod.score(X_train, y_train)

#Predict from manufactured test and score
RFpred = RFmod.predict(X_test)
print('Accuracy score:', accuracy_score(y_test,RFpred)*100)
print('F1 score:', f1_score(y_test, RFpred)*100)
print('Recall score:', recall_score(y_test, RFpred)*100)
print('Precision score:', precision_score(y_test, RFpred)*100)
print('Confusion matrix:', confusion_matrix(y_test, RFpred))
print('Classification report:', classification_report(y_test, RFpred))

#Determine how RF weighted features.
featurelist=list(strokedf_train.columns[:])
plt.bar(range(len(RFmod.feature_importances_)), RFmod.feature_importances_)
plt.xlabel("feature")
plt.ylabel("importance")
plt.title("feature importance")
plt.xticks(range(len(RFmod.feature_importances_)), featurelist,rotation='vertical')
plt.show()
print(featurelist)

#Predict stroke in case study using RF.
RFprestest = RFmod.predict(strokedf_test)
RFpredf = pd.DataFrame(RFprestest,columns=['prediction'])
print(len(strokedf_test))
RFpredf['prediction'].value_counts()

kfold = model_selection.KFold(n_splits=10, random_state=77)
cv_LoR = model_selection.cross_val_score(LogisticRegression(max_iter=10000), X_train, y_train, cv=kfold, scoring='accuracy')
cv_DT = model_selection.cross_val_score(DecisionTreeClassifier(criterion='gini', random_state = 77,max_depth=11), X_train, y_train, cv=kfold, scoring='accuracy')
cv_RF = model_selection.cross_val_score(RandomForestClassifier(n_estimators=100), X_train, y_train, cv=kfold, scoring='accuracy')
results=[cv_LoR,cv_DT,cv_RF]
names=["LoR","DT","RF"]
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

#Curated approach (v1), repeating cleaning from naive approach v1
strokedatacur_train=pd.read_csv('train_2v.csv')
strokedatacur_test=pd.read_csv('test_2v.csv')
strokedatadfcur_train=strokedatacur_train.drop('id',axis=1)
strokedatadfcur_test=strokedatacur_test.drop('id',axis=1)
strokedatadfcur_train = strokedatadfcur_train[strokedatadfcur_train['gender'] != "Other"]
strokedatadfcur_test = strokedatadfcur_test[strokedatadfcur_test['gender'] != "Other"]
strokedatadfcur_train = strokedatadfcur_train[strokedatadfcur_train['age'] > 16]
strokedatadfcur_test = strokedatadfcur_test[strokedatadfcur_test['age'] > 16]
strokedatadfcur_train['bmi'].fillna(strokedatadfcur_train['bmi'].median(), inplace= True)
strokedatadfcur_test['bmi'].fillna(strokedatadfcur_test['bmi'].median(), inplace= True)
strokedatadfcur_train['smoking_status'].fillna(value='unknown',inplace=True)
strokedatadfcur_test['smoking_status'].fillna(value='unknown',inplace=True)
strokedatadfcur_train.isnull().sum()
strokedatadfcur_test.isnull().sum()
strokedatadfcur_train.head(10)

#Going to group age classes becase:
#1) Distribution of ages as seen in earlier graph.
#2) Literature indicates that stroke prevalance doubles every 10 years.
strokedatadfcur_train['age']=strokedatadfcur_train['age'].apply(lambda x:"16-24" if 16<=x<25 else ("25-34" if 25<=x<35 else ("35-44" if 35<=x<44 else ("45-54" if 45<=x<54 else ("55-64" if 55<=x<64 else ("65-74" if 65<=x<74 else ("75-84" if 75<=x<84 else "85+")))))))
strokedatadfcur_test['age']=strokedatadfcur_test['age'].apply(lambda x:"16-24" if 16<=x<25 else ("25-34" if 25<=x<35 else ("35-44" if 35<=x<44 else ("45-54" if 45<=x<54 else ("55-64" if 55<=x<64 else ("65-74" if 65<=x<74 else ("75-84" if 75<=x<84 else "85+")))))))
#Going to group data by average glucose levels because:
#1) Literaure on the subjective partitions the glucose levels after meals as grouped below.
#2) While the distribution shows a clear bimodality I am curious how the model will perform.
strokedatadfcur_train['avg_glucose_level']=strokedatadfcur_train['avg_glucose_level'].apply(lambda x:"healthy" if x<140 else ("pre-diabetes" if 140<=x<200 else "diabetes"))
strokedatadfcur_test['avg_glucose_level']=strokedatadfcur_test['avg_glucose_level'].apply(lambda x:"healthy" if x<140 else ("pre-diabetes" if 140<=x<200 else "diabetes"))
#Going to group data by average glucose levels because:
#1) Literaure on the subjective partitions the BMI as grouped below.
#2) While the distribution shows only a slight right skewness perhaps redistributing as below will introduce symmetry to the distribution.
#3) I am curious on how it will perform.
strokedatadfcur_train['bmi']=strokedatadfcur_train['bmi'].apply(lambda x:"<20" if x<20 else ("20-22" if 20<=x<23 else ("23-24" if 23<=x<25 else ("25-26" if 25<=x<27 else ("27-29" if 27<=x<30 else ("30-34" if 30<=x<35 else "35+"))))))
strokedatadfcur_test['bmi']=strokedatadfcur_test['bmi'].apply(lambda x:"<20" if x<20 else ("20-22" if 20<=x<23 else ("23-24" if 23<=x<25 else ("25-26" if 25<=x<27 else ("27-29" if 27<=x<30 else ("30-34" if 30<=x<35 else "35+"))))))
strokedatadfcur_train.head(10)
sns.catplot(x='age',kind='count',col='stroke',data=strokedatadfcur_train, order=['16-24','25-34','35-44','45-54','55-64','65-74','75-84','85+'])
sns.catplot(x='avg_glucose_level',kind='count',col='stroke',data=strokedatadfcur_train,  order=['healthy','pre-diabetes', 'diabetes'])
sns.catplot(x='bmi',kind='count',col='stroke',data=strokedatadfcur_train,  order=['<20','20-22', '23-24','25-26','27-29','30-34','35+'])

#Encoding data
strokedatadfcur_train['gender'] = enc.fit_transform(strokedatadfcur_train['gender'])
strokedatadfcur_train['ever_married'] = enc.fit_transform(strokedatadfcur_train['ever_married'])
strokedatadfcur_train['work_type'] = enc.fit_transform(strokedatadfcur_train['work_type'])
strokedatadfcur_train['Residence_type'] = enc.fit_transform(strokedatadfcur_train['Residence_type'])
strokedatadfcur_train['smoking_status'] = enc.fit_transform(strokedatadfcur_train['smoking_status'])
strokedatadfcur_train['age'] = enc.fit_transform(strokedatadfcur_train['age'])
strokedatadfcur_train['bmi'] = enc.fit_transform(strokedatadfcur_train['bmi'])
strokedatadfcur_train['avg_glucose_level'] = enc.fit_transform(strokedatadfcur_train['avg_glucose_level'])
strokedatadfcur_test['gender'] = enc.fit_transform(strokedf_test['gender'])
strokedatadfcur_test['ever_married'] = enc.fit_transform(strokedatadfcur_test['ever_married'])
strokedatadfcur_test['work_type'] = enc.fit_transform(strokedatadfcur_test['work_type'])
strokedatadfcur_test['Residence_type'] = enc.fit_transform(strokedatadfcur_test['Residence_type'])
strokedatadfcur_test['smoking_status'] = enc.fit_transform(strokedatadfcur_test['smoking_status'])
strokedatadfcur_test['age'] = enc.fit_transform(strokedatadfcur_test['age'])
strokedatadfcur_test['bmi'] = enc.fit_transform(strokedatadfcur_test['bmi'])
strokedatadfcur_test['avg_glucose_level'] = enc.fit_transform(strokedatadfcur_test['avg_glucose_level'])

#Resampling data (as described earlier)
X_resampcur, y_resampcur=smote.fit_resample(strokedatadfcur_train.loc[:,strokedatadfcur_train.columns!='stroke'],strokedatadfcur_train['stroke'])
print(format(X_resampcur.shape))
print(format(y_resampcur.shape))
y_resamp.value_counts()
X_traincur, X_testcur, y_traincur, y_testcur = train_test_split(X_resampcur, y_resampcur, test_size=0.2, random_state=77)
RFmod=RandomForestClassifier(n_estimators=100,random_state=77)
RFmod.fit(X_traincur,y_traincur)
RFmod.score(X_traincur,y_traincur)
RFpred = RFmod.predict(X_testcur)
print('Accuracy score:', accuracy_score(y_testcur,RFpred)*100)
print('F1 score:', f1_score(y_testcur, RFpred)*100)
print('Recall score:', recall_score(y_testcur, RFpred)*100)
print('Precision score:', precision_score(y_testcur, RFpred)*100)
print('Confusion matrix:', confusion_matrix(y_testcur, RFpred))
print('Classification report:', classification_report(y_testcur, RFpred))
featurelist=list(strokedatadfcur_train.columns[:])
plt.bar(range(len(RFmod.feature_importances_)), RFmod.feature_importances_)
plt.xlabel("feature")
plt.ylabel("importance")
plt.title("feature importance")
plt.xticks(range(len(RFmod.feature_importances_)), featurelist,rotation='vertical')
plt.show()
print(featurelist)
RFprestest = RFmod.predict(strokedatadfcur_test)
RFpredf = pd.DataFrame(RFprestest,columns=['prediction'])
print(len(strokedatadfcur_test))
RFpredf['prediction'].value_counts()


###Curated approach v2, repeating cleaning from curatved v1 with exceptions
strokedatacur_train=pd.read_csv('train_2v.csv')
strokedatacur_test=pd.read_csv('test_2v.csv')
strokedatadfcur_train=strokedatacur_train.drop('id',axis=1)
strokedatadfcur_test=strokedatacur_test.drop('id',axis=1)
strokedatadfcur_train = strokedatadfcur_train[strokedatadfcur_train['gender'] != "Other"]
strokedatadfcur_test = strokedatadfcur_test[strokedatadfcur_test['gender'] != "Other"]
strokedatadfcur_train = strokedatadfcur_train[strokedatadfcur_train['age'] > 16]
strokedatadfcur_test = strokedatadfcur_test[strokedatadfcur_test['age'] > 16]
strokedatadfcur_train['bmi'].fillna(strokedatadfcur_train['bmi'].median(), inplace= True)
strokedatadfcur_test['bmi'].fillna(strokedatadfcur_test['bmi'].median(), inplace= True)
strokedatadfcur_train['smoking_status'].fillna(value='unknown',inplace=True)
strokedatadfcur_test['smoking_status'].fillna(value='unknown',inplace=True)
strokedatadfcur_train['age']=strokedatadfcur_train['age'].apply(lambda x:"16-24" if 16<=x<25 else ("25-34" if 25<=x<35 else ("35-44" if 35<=x<44 else ("45-54" if 45<=x<54 else ("55-64" if 55<=x<64 else ("65-74" if 65<=x<74 else ("75-84" if 75<=x<84 else "85+")))))))
strokedatadfcur_test['age']=strokedatadfcur_test['age'].apply(lambda x:"16-24" if 16<=x<25 else ("25-34" if 25<=x<35 else ("35-44" if 35<=x<44 else ("45-54" if 45<=x<54 else ("55-64" if 55<=x<64 else ("65-74" if 65<=x<74 else ("75-84" if 75<=x<84 else "85+")))))))
strokedatadfcur_train['gender'] = enc.fit_transform(strokedatadfcur_train['gender'])
strokedatadfcur_train['ever_married'] = enc.fit_transform(strokedatadfcur_train['ever_married'])
strokedatadfcur_train['work_type'] = enc.fit_transform(strokedatadfcur_train['work_type'])
strokedatadfcur_train['Residence_type'] = enc.fit_transform(strokedatadfcur_train['Residence_type'])
strokedatadfcur_train['smoking_status'] = enc.fit_transform(strokedatadfcur_train['smoking_status'])
strokedatadfcur_train['age'] = enc.fit_transform(strokedatadfcur_train['age'])
strokedatadfcur_test['gender'] = enc.fit_transform(strokedf_test['gender'])
strokedatadfcur_test['ever_married'] = enc.fit_transform(strokedatadfcur_test['ever_married'])
strokedatadfcur_test['work_type'] = enc.fit_transform(strokedatadfcur_test['work_type'])
strokedatadfcur_test['Residence_type'] = enc.fit_transform(strokedatadfcur_test['Residence_type'])
strokedatadfcur_test['smoking_status'] = enc.fit_transform(strokedatadfcur_test['smoking_status'])
strokedatadfcur_test['age'] = enc.fit_transform(strokedatadfcur_test['age'])

X_resampcur, y_resampcur=smote.fit_resample(strokedatadfcur_train.loc[:,strokedatadfcur_train.columns!='stroke'],strokedatadfcur_train['stroke'])
print(format(X_resampcur.shape))
print(format(y_resampcur.shape))
y_resamp.value_counts()
strokedatadfcur_test.info()
X_traincur, X_testcur, y_traincur, y_testcur = train_test_split(X_resampcur, y_resampcur, test_size=0.2, random_state=77)
RFmod=RandomForestClassifier(n_estimators=100,random_state=77)
RFmod.fit(X_traincur,y_traincur)
RFmod.score(X_traincur,y_traincur)
RFpred = RFmod.predict(X_testcur)
print('Accuracy score:', accuracy_score(y_testcur,RFpred)*100)
print('F1 score:', f1_score(y_testcur, RFpred)*100)
print('Recall score:', recall_score(y_testcur, RFpred)*100)
print('Precision score:', precision_score(y_testcur, RFpred)*100)
print('Confusion matrix:', confusion_matrix(y_testcur, RFpred))
print('Classification report:', classification_report(y_testcur, RFpred))
featurelist=list(strokedatadfcur_train.columns[:])
plt.bar(range(len(RFmod.feature_importances_)), RFmod.feature_importances_)
plt.xlabel("feature")
plt.ylabel("importance")
plt.title("feature importance")
plt.xticks(range(len(RFmod.feature_importances_)), featurelist,rotation='vertical')
plt.show()
print(featurelist)
RFprestest = RFmod.predict(strokedatadfcur_test)
RFpredf = pd.DataFrame(RFprestest,columns=['prediction'])
print(len(strokedatadfcur_test))
RFpredf['prediction'].value_counts()
