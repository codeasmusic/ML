# coding=utf-8

from sklearn.model_selection import GridSearchCV

def tuning_params(clf, params, x, y, cv=3):
    # params = {
    #     'learning_rate': [0.01, 0.05, 0.1],
    #     'n_estimators': range(100, 501, 100)
    # }

    gs = GridSearchCV(estimator=clf, param_grid=params, n_jobs=-1, cv=cv)
    gs.fit(x, y)

    print "mean score: ", gs.cv_results_['mean_test_score']
    print 'best score: ', gs.best_score_

    print 'best parameters:'
    for p in params.keys():
        print p, gs.best_params_[p]

