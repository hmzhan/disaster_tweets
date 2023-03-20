import numpy as np
import matplotlib.pyplot as plt

from sklearn import set_config
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

from sksurv.datasets import load_flchain, load_gbsg2, load_veterans_lung_cancer
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score,
)
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder, encode_categorical


class Brier:
    def __init__(self):
        self.gbsg_X, self.gbsg_y = load_gbsg2()
        self.gbsg_X = encode_categorical(self.gbsg_X)
        self.gbsg_X_train, self.gbsg_X_test, self.gbsg_y_train, self.gbsg_y_test = train_test_split(
            self.gbsg_X, self.gbsg_y, stratify=self.gbsg_y["cens"], random_state=1
        )


