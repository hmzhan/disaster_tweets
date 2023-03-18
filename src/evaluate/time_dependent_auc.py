import numpy as np
from sksurv.datasets import load_flchain
from sksurv.metrics import cumulative_dynamic_auc, concordance_index_ipcw
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt


class Data:
    def __init__(self):
        self.x = None
        self.y = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.x, self.y = load_flchain()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y,
                                                                                test_size=0.2, random_state=0)

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

