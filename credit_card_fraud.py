# -*- coding: utf-8 -*-
"""credit_card_fraud.ipynb
"""

import pandas as pd

data=pd.read_csv('C:\Users\ASUS\Desktop\Project\DataSet_CreditCard.csv')
data.head()

s=data.corr()
s['Class'].sort_values(ascending=False)

import matplotlib.pyplot as plt
import seaborn as sns
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(s, vmax=1, square=True);
plt.show()

data.isnull().values.any()

fraud=data[data['Class']==1]
fraud.shape

fraud.describe()

normal=data[data['Class']==0]
normal.shape

normal.describe()

print('% of fraud is',(len(fraud)/float(len(normal))))

data.hist(figsize=(20,20))
plt.show()

X = data.drop(['Class'],axis=1)
y = data['Class']

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score,f1_score,recall_score,classification_report,average_precision_score
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=10)
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
log_reg=LogisticRegression()
log_reg.fit(X_train,y_train.values.ravel())
log_pred=log_reg.predict(X_test)

print('Accuracy Score:{}'.format(accuracy_score(y_test, log_pred)))
print('f1 Score:{}'.format(f1_score(y_test, log_pred)))
print('Recall Score:{}'.format(recall_score(y_test,log_pred)))
print('Classification Report:')
print(classification_report(y_test,log_pred))
print('Average Precison-Recall Score:{}'.format(average_precision_score(y_test,log_pred)))

from sklearn.svm import SVC
svm=SVC()
svm.fit(X_train,y_train.values.ravel())
svm_pred=svm.predict(X_test)

print('Accuracy Score:{}'.format(accuracy_score(y_test, svm_pred)))
print('f1 Score:{}'.format(f1_score(y_test, svm_pred)))
print('Recall Score:{}'.format(recall_score(y_test,svm_pred)))
print('Classification Report:')
print(classification_report(y_test,svm_pred))
print('Average Precison-Recall Score:{}'.format(average_precision_score(y_test,svm_pred)))

from sklearn.tree import DecisionTreeClassifier
dst=DecisionTreeClassifier()
dst.fit(X_train,y_train)
dst_pred=dst.predict(X_test)

print('Accuracy Score:{}'.format(accuracy_score(y_test, dst_pred)))
print('f1 Score:{}'.format(f1_score(y_test, dst_pred)))
print('Recall Score:{}'.format(recall_score(y_test,dst_pred)))
print('Classification Report:')
print(classification_report(y_test,dst_pred))
print('Average Precison-Recall Score:{}'.format(average_precision_score(y_test,dst_pred)))

from sklearn.ensemble import RandomForestClassifier
rmf=RandomForestClassifier(n_estimators=100)
rmf.fit(X_train,y_train)
rmf_pred=rmf.predict(X_test)
print('Accuracy Score:{}'.format(accuracy_score(y_test, rmf_pred)))
print('f1 Score:{}'.format(f1_score(y_test, rmf_pred)))
print('Recall Score:{}'.format(recall_score(y_test,rmf_pred)))
print('Classification Report:')
print(classification_report(y_test,rmf_pred))
print('Average Precison-Recall Score:{}'.format(average_precision_score(y_test,rmf_pred)))

from sklearn.ensemble import AdaBoostClassifier
abc=AdaBoostClassifier(n_estimators=100)
abc.fit(X_train,y_train)
abc_pred=abc.predict(X_test)
print('Accuracy Score:{}'.format(accuracy_score(y_test, abc_pred)))
print('f1 Score:{}'.format(f1_score(y_test, abc_pred)))
print('Recall Score:{}'.format(recall_score(y_test,abc_pred)))
print('Classification Report:')
print(classification_report(y_test,abc_pred))
print('Average Precison-Recall Score:{}'.format(average_precision_score(y_test,abc_pred)))

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
kfold = model_selection.KFold(n_splits=10, random_state=7,shuffle=True)
num_trees = 100
model = BaggingClassifier(base_estimator=dst, n_estimators=num_trees, random_state=7)
model.fit(X_train,y_train)
dstbc_pred=model.predict(X_test)
print('Accuracy Score:{}'.format(accuracy_score(y_test, dstbc_pred)))
print('f1 Score:{}'.format(f1_score(y_test, dstbc_pred)))
print('Recall Score:{}'.format(recall_score(y_test,dstbc_pred)))
print('Classification Report:')
print(classification_report(y_test,dstbc_pred))
print('Average Precison-Recall Score:{}'.format(average_precision_score(y_test,dstbc_pred)))

from sklearn import model_selection
from sklearn.ensemble import BaggingClassifier
kfold = model_selection.KFold(n_splits=10, random_state=7,shuffle=True)
num_trees = 100
model_svm = BaggingClassifier(base_estimator=svm, n_estimators=num_trees, random_state=7)
model_svm.fit(X_train,y_train)
svmbc_pred=model.predict(X_test)
print('Accuracy Score:{}'.format(accuracy_score(y_test, svmbc_pred)))
print('f1 Score:{}'.format(f1_score(y_test, svmbc_pred)))
print('Recall Score:{}'.format(recall_score(y_test,svmbc_pred)))
print('Classification Report:')
print(classification_report(y_test,svmbc_pred))
print('Average Precison-Recall Score:{}'.format(average_precision_score(y_test,svmbc_pred)))

