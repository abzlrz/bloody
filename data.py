import glob, os, sys
import numpy as np 
from sklearn.model_selection import GridSearchCV

from extraction import build_features

def build_dataset(path):
    x = glob.glob(os.path.join(path, '1\\*.jpg'))
    y = glob.glob(os.path.join(path, '0\\*.jpg'))

    _x = build_features(x, 1)
    _y = build_features(y, 0)

    db = _x + _y
    return db


def validate_model(model, params, train_data, test_data):
    clf = GridSearchCV(estimator=model, param_grid=params, n_jobs=3, cv=10)
    clf.fit(train_data[0], train_data[1])

    grid_scores = clf.grid_scores_
    best_scores = clf.best_score_
    best_params = clf.best_params_
    test_score = clf.score(test_data[0], test_data[1])

    return grid_scores, best_scores, best_params, test_score

