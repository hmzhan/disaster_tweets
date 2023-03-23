import numpy as np
import pandas as pd
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
        self.cph_gbsg = CoxnetSurvivalAnalysis(l1_ratio=0.99, fit_baseline_model=True)
        self.rsf_gbsg = RandomSurvivalForest(max_depth=2, random_state=1)
        self.score_cindex = None
        self.score_brier = None

    def cox_model(self):
        self.cph_gbsg.fit(self.gbsg_X_train, self.gbsg_y_train)

    def rsf_model(self):
        self.rsf_gbsg.fit(self.gbsg_X_train, self.gbsg_y_train)

    def calculate_cindex(self):
        self.score_cindex = pd.Series(
            [
                self.rsf_gbsg.score(self.gbsg_X_test, self.gbsg_y_test),
                self.cph_gbsg.score(self.gbsg_X_test, self.gbsg_y_test),
                0.5,
            ],
            index=["RSF", "CPH", "Random"], name="c-index",
        )
        self.score_cindex.round(3)

    def calculate_ibs(self):
        lower, upper = np.percentile(self.gbsg_y["time"], [10, 90])
        gbsg_times = np.arange(lower, upper + 1)

        rsf_surv_prob = np.row_stack([
            fn(gbsg_times)
            for fn in self.rsf_gbsg.predict_survival_function(self.gbsg_X_test)
        ])
        cph_surv_prob = np.row_stack([
            fn(gbsg_times)
            for fn in self.cph_gbsg.predict_survival_function(self.gbsg_X_test)
        ])
        random_surv_prob = 0.5 * np.ones(
            (self.gbsg_y_test.shape[0], gbsg_times.shape[0])
        )

        self.score_brier = pd.Series(
            [
                integrated_brier_score(self.gbsg_y, self.gbsg_y_test, prob, gbsg_times)
                for prob in (rsf_surv_prob, cph_surv_prob, random_surv_prob)
            ],
            index=["RSF", "CPH", "Random"],
            name="IBS"
        )
