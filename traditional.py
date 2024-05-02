"""Accepts a Scikit-learn (sklearn in code) model object, loads data to train and evaluate this model
by Jason Liu"""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
# list of models we need to make exceptions for
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
# typing utility
from typing import Protocol
from numpy.typing import NDArray
from sklearn.utils._testing import ignore_warnings # in new version, it's _testing, NOT testing!
# see https://stackoverflow.com/questions/62336142/modulenotfounderror-no-module-named-sklearn-utils-testing
from sklearn.exceptions import ConvergenceWarning
# linear models require feature scaling; make_pipeline()
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# a 'typing dummy' using structural typing! https://stackoverflow.com/questions/54868698/what-type-is-a-sklearn-model
class SklearnModel(Protocol):
    """Simulates the interface of Scikit-learn models"""
    def fit(self, incoming_train, labels_train, sample_weight=None): 
        """update model parameters using incoming training examples and label"""
        ... # the purpose of this class is to 
    def predict(self, incoming_test) -> NDArray: 
        """make prediction from testing data"""
        ...
    def score(self, incoming_test, labels_test, sample_weight=None) -> NDArray: 
        """get average accuracy from testing examples and label"""
        ...
    def predict_proba(self, labels_test : NDArray) -> NDArray:
        """get probability for each example being categorized to each class""" 
        ...
# Alternatively, sklearn's 'Pipeline' object also works as a 'typing dummy'

# sklearn can algorithm-not-converging warning, which is not always useful e.g. 
#   when we know the algorithm isn't converging; this decorator hides those warnings
# https://stackoverflow.com/questions/53784971/how-to-disable-convergencewarning-using-sklearn
@ignore_warnings(category=ConvergenceWarning) # works despite Intellisense' typing complaints
def cross_validation(sklearn_classifier : SklearnModel, 
    x_cv : NDArray, y_cv : NDArray, 
    train_indices : NDArray, valid_indices : NDArray, # shape=[fold, ...]
    ):
    """Accepts a sklearn model, loading data to train and evaluate it \n
    Inputs:
        1 ```sklearn_classifier``` (SkleranModel): a Scikit-learn object (initialized with parameters)\n"""
    if type(sklearn_classifier) == LogisticRegression():
        # make_pipeline() combines different models into a Pipeline object that
        #   has interface similar to a sklearn model
        sklearn_classifier = make_pipeline(StandardScaler(), LogisticRegression())
    elif type(sklearn_classifier) == MLPClassifier(): # sklearn's neural network does NOT support GPU
        sklearn_classifier = make_pipeline(StandardScaler(), MLPClassifier())
    else:
        pass # no pipeline needed!
    classifiers: list[SklearnModel] = []
    AUROCies = []
    AUPRCies = []
    for fold in range(train_indices.shape[0]):
        print(f"\tfold {fold + 1}:", end=" ")
        sklearn_classifier.fit(x_cv[train_indices[fold]], y_cv[train_indices[fold]])
        # preformance metrics below! Some metrics use scores, not predictions!
        scores_valid = sklearn_classifier.predict_proba(x_cv[valid_indices[fold]])[:, 1]
        # print(f"scores_test shape: {scores_test.shape}")
        ROC_AUC = roc_auc_score(y_true=y_cv[valid_indices[fold]], y_score=scores_valid)
        AUROCies.append(ROC_AUC)
        AUPRC = average_precision_score(y_true=y_cv[valid_indices[fold]], y_score=scores_valid)
        AUPRCies.append(AUPRC)
        classifiers.append(sklearn_classifier)
    mean_AUROC = np.mean(AUROCies)
    std_AUROC = np.std(AUPRCies)
    mean_AUPRC = np.mean(AUPRCies)
    std_AUPRC = np.std(AUPRCies)
    print()
    print("TRAIN-AND-VALIDATE",
        f"\n\tAUROC:      mean {mean_AUROC:.4f}",
        f"\n\t            std  {std_AUROC:.4f}",
        f"\n\tAUPRC:      mean {mean_AUPRC:.4f}",
        f"\n\t            std  {std_AUPRC:.4f}",)
    # currently, I use AUROC to choose best classifier
    best_index = AUPRCies.index(max(AUPRCies))
    print(f"Best classifier is in fold: {best_index}")
    return classifiers[best_index]
    
        
