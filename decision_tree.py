#! /usr/bin/env python

from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
import sys
sys.path.append('path for the function ')
tree = DecisionTreeClassifier(criterion='entropy',max_depth=3,random_state=0)
tree = fit(x_train,y_train)# No standardised
x_combined = np.vstack((x_train,x_test))
y_combined = np.hstack((y_train,y_test))

plot_decision_regions(x_combined,y_combined,classifier=tree,test_idx=range(105,150))

plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]')
plt.legend(loc='upper left')
plt.show()

from sklearn.tree import export_graphviz
export_graphviz(tree,out_file='tree.dot',feature_names=['petal length','petal width'])


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy',n_estimators = 10,random_state=1,n_jobs=2)

forest.fit(x_train,y_train)

plot_decision_regions(x_combined,y_combined,classifier=forest,test_idx = range(1055,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc = 'upper left')
plt.show()

