import numpy as np
import pandas as pd

data=pd.read_csv("traindata.csv", header= None)
#print(data)
label=pd.read_csv("trainlabel.csv", header= None)
#print(label)
test=pd.read_csv("testdata.csv", header= None)
#print(test)

X=data.values
y=label.values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import Perceptron
ppn = Perceptron(n_iter=40, eta0=0.1, random_state=0)
pipe_ppn = Pipeline([('scl', StandardScaler()),
                     ('clf', ppn)])

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('clf', lr)])

from sklearn.svm import SVC
svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
pipe_svm = Pipeline([('scl', StandardScaler()),
                     ('clf', svm)])

from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
pipe_tree = Pipeline([('scl', StandardScaler()),
                      ('clf', tree)])

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',
                                n_estimators=10, 
                                random_state=1,
                                n_jobs=2)
pipe_forest = Pipeline([('scl', StandardScaler()),
                        ('clf', forest)])

pipes = [pipe_ppn, pipe_lr, pipe_svm, pipe_tree, pipe_forest]

for pipe in pipes:
        pipe.fit(X_train, y_train)
        score = pipe.score(X_test, y_test)
        print(score)

prediction = pipe_tree.predict(test.values)
#print(prediction)
prediction = pd.DataFrame(prediction)
prediction.to_csv("project1_20461901.csv", header=False, index=False)
