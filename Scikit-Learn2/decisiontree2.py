from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from IPython.display import Image
import pydot
import pydotplus
import os

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

cancer = load_breast_cancer()

X_train, X_test, y_train, y_test = train_test_split(
cancer.data, cancer.target, stratify=cancer.target, random_state=42)

dTree = DecisionTreeClassifier(criterion='entropy')

dTree = dTree.fit(X_train, y_train)

print("Train Set Score1 : {:.2f}".format(dTree.score(X_train, y_train)))
print("Test  Set Score1 : {:.2f}".format(dTree.score(X_test, y_test)))

graph1_data = export_graphviz(dTree, out_file= None, class_names=["malignant","benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
graph1 = pydotplus.graph_from_dot_data(graph1_data)
Image(graph1.create_png())

graph1.write_png("graph1.png")


dTreeAll = DecisionTreeClassifier()

dTreeAll = dTreeAll.fit(X_train, y_train)

print("Train Set Score2 : {:.2f}".format(dTreeAll.score(X_train, y_train)))
print("Test  Set Score2 : {:.2f}".format(dTreeAll.score(X_test, y_test)))

graph2_data = export_graphviz(dTreeAll, out_file= None, class_names=["malignant","benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
graph2 = pydotplus.graph_from_dot_data(graph2_data)
Image(graph2.create_png())

graph2.write_png("graph2.png")


dTreeLimit = DecisionTreeClassifier(max_depth=2, random_state=0)

dTreeLimit = dTreeLimit.fit(X_train, y_train)

print("Train Set Score3 : {:.2f}".format(dTreeLimit.score(X_train, y_train)))
print("Test  Set Score3 : {:.2f}".format(dTreeLimit.score(X_test, y_test)))


graph3_data = export_graphviz(dTreeLimit, out_file= None, class_names=["malignant","benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
graph3 = pydotplus.graph_from_dot_data(graph3_data)
Image(graph3.create_png())

graph3.write_png("graph3.png")

dTreeLimit2 = DecisionTreeClassifier(max_depth=3, random_state=0)

dTreeLimit2 = dTreeLimit2.fit(X_train, y_train)

print("Train Set Score4 : {:.2f}".format(dTreeLimit2.score(X_train, y_train)))
print("Test  Set Score4 : {:.2f}".format(dTreeLimit2.score(X_test, y_test)))


graph4_data = export_graphviz(dTreeLimit2, out_file= None, class_names=["malignant","benign"],
                feature_names=cancer.feature_names, impurity=False, filled=True)
graph4 = pydotplus.graph_from_dot_data(graph4_data)
Image(graph4.create_png())

graph4.write_png("graph4.png")
