from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from __future__ import division

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC


class FeatureSelection:


    classifiers =[
        BernoulliNB(),
        MultinomialNB(),
        GaussianNB(),
        DecisionTreeClassifier(),
        RandomForestClassifier(n_estimators=10),
        OneVsRestClassifier(LinearSVC(random_state=0)),
        OneVsRestClassifier(LogisticRegression()),
        OneVsRestClassifier(SGDClassifier()),
        OneVsRestClassifier(RidgeClassifier()),
    ]


    def univariateFeatureSelection(self,X,y,nfolds,clf,nfeats,clfname,scoreFunc):
        kfold = KFold(X.shape[0],n_folds=nfolds)
        acc = 0
        i = 0
        print("%s (#-features = %d).... %"% (clfname,nfeats))
        for train,test in kfold:
            i += 1
            X_train,X_test,y_train,y_test =  X[train],X[test],y[train],y[test]
            clf.fit(X_train,y_train)




