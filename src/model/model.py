import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sksurv.datasets import load_gbsg2
from sksurv.preprocessing import encode_categorical
from sksurv.ensemble import RandomSurvivalForest

from sksurv.metrics import (
    as_concordance_index_ipcw_scorer,
    as_cumulative_dynamic_auc_scorer,
    as_integrated_brier_score_scorer
)

import matplotlib.pyplot as plt


class Model:
    def __init__(self):
        self.gbsg_X, self.gbsg_y = load_gbsg2()
        self.gbsg_X = encode_categorical(self.gbsg_X)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.gbsg_X, self.gbsg_y, test_size=0.25, random_state=20
        )
        self.cv = KFold(n_splits=3, shuffle=True, random_state=1)
        self.cv_param_grid = {"estimator__max_depth": np.arange(1, 10, dtype=int)}
        self.rsf_gbsg = RandomSurvivalForest(max_depth=2, random_state=1)
        self.gcv_cindex = None
        self.gcv_iauc = None
        self.gcv_ibs = None
        self.rsf = None

    def run_cv(self):
        lower, upper = np.percentile(self.gbsg_y["time"], [10, 90])
        gbsg_times = np.arange(lower, upper + 1)

        self.gcv_cindex = GridSearchCV(
            as_concordance_index_ipcw_scorer(self.rsf_gbsg, tau=gbsg_times[-1]),
            param_grid=self.cv_param_grid,
            cv=self.cv,
            n_jobs=4
        ).fit(self.gbsg_X, self.gbsg_y)

        self.gcv_iauc = GridSearchCV(
            as_cumulative_dynamic_auc_scorer(self.rsf_gbsg, times=gbsg_times),
            param_grid=self.cv_param_grid,
            cv=self.cv,
            n_jobs=4
        ).fit(self.gbsg_X, self.gbsg_y)

        self.gcv_ibs = GridSearchCV(
            as_integrated_brier_score_scorer(self.rsf_gbsg, times=gbsg_times),
            param_grid=self.cv_param_grid,
            cv=self.cv,
            n_jobs=4
        ).fit(self.gbsg_X, self.gbsg_y)

    @staticmethod
    def _plot_grid_search_results(gcv, ax, name):
        ax.errorbar(
            x=gcv.cv_results_["param_estimator__max_depth"].filled(),
            y=gcv.cv_results_["mean_test_score"],
            yerr=gcv.cv_results_["std_test_score"],
        )
        ax.plot(
            gcv.best_params_["estimator__max_depth"],
            gcv.best_score_,
            'ro',
        )
        ax.set_ylabel(name)
        ax.yaxis.grid(True)

    def plot_results(self):
        _, axs = plt.subplots(3, 1, figsize=(6, 6), sharex=True)
        axs[-1].set_xlabel("max_depth")
        self._plot_grid_search_results(self.gcv_cindex, axs[0], "c-index")
        self._plot_grid_search_results(self.gcv_iauc, axs[1], "iAUC")
        self._plot_grid_search_results(self.gcv_ibs, axs[2], "$-$IBS")

    def rsf_model(self):
        self.rsf = RandomSurvivalForest(
            n_estimators=1000,
            min_samples_split=10,
            min_samples_leaf=15,
            n_jobs=-1,
            random_state=20
        )
        self.rsf.fit(self.X_train, self.y_train)
        print(self.rsf.score(self.X_test, self.y_test))

    def make_test_data(self):
        X_test_sorted = self.X_test.sort_values(by=["pnodes", "age"])
        return pd.concat([X_test_sorted.head(3), X_test_sorted.tail(3)])

    def plot_survival(self):
        X_test_sel = self.make_test_data()
        surv = self.rsf.predict_survival_function(X_test_sel, return_array=True)
        for i, s in enumerate(surv):
            plt.step(self.rsf.event_times_, s, where="post", label=str(i))
        plt.ylabel("Survival probability")
        plt.xlabel("Time in day")
        plt.legend()
        plt.grid(True)

    def plot_chf(self):
        X_test_sel = self.make_test_data()
        chf = self.rsf.predict_cumulative_hazard_function(X_test_sel, return_array=True)
        for i, s in enumerate(chf):
            plt.step(self.rsf.event_times_, s, where="post", label=str(i))
        plt.ylabel("Cumulative hazard")
        plt.xlabel("Time in day")
        plt.legend()
        plt.grid(True)
