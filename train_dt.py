import sklearn
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
import pandas as pd

def main():
    X = pd.read_csv('./data/origin/model1.csv')
    y = X["P[t]"]
    X.drop(labels=['P[t]'], axis=1, inplace=True)

    model = DecisionTreeClassifier()
    params = {'criterion':['gini', 'entropy'], 'max_depth': range(2,8), 'min_samples_split': range(2, 10)}
    gs = GridSearchCV(estimator=model, param_grid=params, scoring='accuracy', cv=5, verbose=2)
    gs.fit(X,y)
    print gs.best_params_, gs.best_score_

    model = DecisionTreeClassifier(**gs.best_params_)
    model.fit(X,y)

    # from sklearn.externals.six import StringIO
    import pydotplus
    dot_data = tree.export_graphviz(model, out_file=None,
                         feature_names=['IAT[t]','IAT[t-1]',
                                        'BD[t]','BD[t-1]',
                                        'IATC[t]','IATC[t-1]',
                                        'BDC[t]','BDC[t-1]'],
                         class_names=['0', '1'],
                         filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data)
    graph.write_pdf("dt.pdf")


if __name__ == "__main__":
    main()