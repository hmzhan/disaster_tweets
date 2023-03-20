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
from sksurv.preprocessing import OneHotEncoder


class TimeDependentAUC:
    def __init__(self):
        self.x, self.y = load_flchain()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=0.2,
                                                                                random_state=0)

    def impute_data(self, num_columns):
        imputer = SimpleImputer().fit(self.x_train.loc[:, num_columns])
        x_test_imputed = imputer.transform(self.x_test.loc[:, num_columns])
        return x_test_imputed

    def plot_cumulative_dynamic_auc(self, risk_score, label, color=None):
        times = np.percentile(self.y["futime"], np.linspace(5, 81, 15))
        auc, mean_auc = cumulative_dynamic_auc(self.y_train, self.y_test, risk_score, times)
        plt.plot(times, auc, marker="o", color=color, label=label)
        plt.xlabel("days from enrollment")
        plt.ylabel("time-dependent auc")
        plt.axhline(mean_auc, color=color, linestyle="--")
        plt.legend()


class EvaluateModel:
    def __init__(self):
        self.va_x, self.va_y = load_veterans_lung_cancer()
        self.va_x_train, self.va_x_test, self.va_y_train, self.va_y_test = train_test_split(
            self.va_x, self.va_y, test_size=0.2, stratify=self.va_y["Status"], random_state=0
        )
        self.va_times = np.arange(8, 184, 7)
        self.cph = None
        self.rsf = None

    def cox_model(self):
        self.cph = make_pipeline(OneHotEncoder(), CoxPHSurvivalAnalysis())
        self.cph.fit(self.va_x_train, self.va_y_train)

    def plot_cox_model_auc(self):
        cph_risk_scores = self.cph.predict(self.va_x_test)
        cph_auc, cph_mean_auc = cumulative_dynamic_auc(
            self.va_y_train, self.va_y_test, cph_risk_scores, self.va_times
        )
        plt.plot(self.va_times, cph_auc, marker="o")
        plt.axhline(cph_mean_auc, linestyle="--")
        plt.xlabel("days from enrollment")
        plt.ylabel("time-dependent AUC")
        plt.grid(True)

    def rsf_model(self):
        self.rsf = make_pipeline(
            OneHotEncoder(),
            RandomSurvivalForest(n_estimators=100, min_samples_leaf=7, random_state=0)
        )
        self.rsf.fit(self.va_x_train, self.va_y_train)

    def plot_model_auc(self):
        cph_risk_scores = self.cph.predict(self.va_x_test)
        cph_auc, cph_mean_auc = cumulative_dynamic_auc(
            self.va_y_train, self.va_y_test, cph_risk_scores, self.va_times
        )

        rsf_chf_funcs = self.rsf.predict_cumulative_hazard_function(
            self.va_x_test, return_array=False
        )
        rsf_risk_scores = np.row_stack([chf(self.va_times) for chf in rsf_chf_funcs])
        rsf_auc, rsf_mean_auc = cumulative_dynamic_auc(
            self.va_y_train, self.va_y_test, rsf_risk_scores, self.va_times
        )
        plt.plot(self.va_times, cph_auc, label="CoxPH (mean AUC = {:.3f})".format(cph_mean_auc))
        plt.plot(self.va_times, rsf_auc, label="RSF (mean AUC = {:.3f})".format(rsf_mean_auc))
        plt.xlabel("days from enrollment")
        plt.ylabel("time-dependent AUC")
        plt.legend(loc="lower center")
        plt.grid(True)
