#  link kham khảo
#https://stackoverflow.com/questions/65682019/attributeerror-str-object-has-no-attribute-decode-in-fitting-logistic-regre
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
from sklearn.model_selection import train_test_split
# import the class
from sklearn.linear_model import LogisticRegression

#import pandas
import pandas as pd
data = pd.read_csv("diabetes.csv")
N = data.shape[0]
a = data.iloc[:, 0]
a = a.values.reshape(-1, 1)
b = data.iloc[:, 1]
b = b.values.reshape(-1, 1)
c = data.iloc[:, 2]
c = c.values.reshape(-1, 1)
d = data.iloc[:, 3]
d = d.values.reshape(-1, 1)
e = data.iloc[:, 4]
e = e.values.reshape(-1, 1)
f = data.iloc[:, 5]
f = f.values.reshape(-1, 1)
g = data.iloc[:, 6]
g = g.values.reshape(-1, 1)
h = data.iloc[:, 7]
h = h.values.reshape(-1, 1)
X = np.hstack((np.ones((N, 1)), a,b,c,d,e,f,g,h))
y = data.iloc[:, 8]
y = y.values.reshape(-1, 1)
y = y.flatten()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
logreg = LogisticRegression(solver='liblinear')
# fit the model with data
logreg.fit(X_train,y_train)
y_pred=logreg.predict(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
plt.figure(1)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')

print("Accuracy:",metrics.accuracy_score(y_test, y_pred))
print("Precision:",metrics.precision_score(y_test, y_pred))
print("Recall:",metrics.recall_score(y_test, y_pred))
plt.figure(2)
y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()