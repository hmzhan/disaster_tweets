import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
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
        self.cv = KFold(n_splits=3, shuffle=True, random_state=1)
        self.cv_param_grid = {"estimator__max_depth": np.arange(1, 10, dtype=int)}
        self.rsf_gbsg = RandomSurvivalForest(max_depth=2, random_state=1)
        self.gcv_cindex = None
        self.gcv_iauc = None
        self.gcv_ibs = None

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


