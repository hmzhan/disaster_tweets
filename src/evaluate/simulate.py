import numpy as np
import pandas as pd
import scipy.optimize as opt
from sksurv.metrics import (
    concordance_index_censored,
    concordance_index_ipcw,
    cumulative_dynamic_auc,
    integrated_brier_score
)
from sksurv.util import Surv


class SimulateSurvival:
    def __init__(self, n_samples, hazard_ratio):
        self.n_samples = n_samples
        self.hazard_ratio = hazard_ratio
        self.baseline_hazard = 0.1
        self.rnd = np.random.RandomState(seed=987)
        self.time_event = None

    def _generate_marker(self):
        """
        Generate markers for simulation study
        :return: simulated data
        """
        X = self.rnd.randn(self.n_samples, 1)
        hazard_ratio = np.array([self.hazard_ratio])
        logits = np.dot(X, np.log(hazard_ratio))

        u = self.rnd.uniform(size=self.n_samples)
        self.time_event = -np.log(u) / (self.baseline_hazard * np.exp(logits))

        X = np.squeeze(X)
        actual = concordance_index_censored(np.ones(
            self.n_samples, dtype=bool),
            self.time_event, X
        )
        return X, actual[0]

    def _get_observed_time(self, x):
        """
        Get observed time
        :param x:
        :return: event and time
        """
        rnd_cens = np.random.RandomState(0)
        time_censor = rnd_cens.uniform(high=x, size=self.n_samples)
        event = self.time_event < time_censor
        time = np.where(event, self.time_event, time_censor)
        return event, time

    def generate_survival_data(self, percentage_cens):
        """
        Generate survival data
        :param percentage_cens:
        :return:
        """
        def _censoring_amount(x):
            event, _ = self._get_observed_time(x)
            cens = 1.0 - event.sum() / event.shape[0]
            return (cens - percentage_cens) ** 2

        X, actual_c = self._generate_marker()
        res = opt.minimize_scalar(
            _censoring_amount,
            method="bounded",
            bounds=(0, self.time_event.max())
        )
        event, time = self._get_observed_time(res.x)

        tau = time[event].max()
        y = Surv.from_arrays(event=event, time=time)
        mask = time < tau
        X_test = X[mask]
        y_test = y[mask]
        return X_test, y_test, y, actual_c

    def simulation(self, n_repeats=100):
        measures = ("censoring", "Harrel's C", "Uno's C",)
        data_mean = {}
        data_std = {}
        for measure in measures:
            data_mean[measure] = []
            data_std[measure] = []

        # iterate over different amount of censoring
        for cens in (.1, .25, .4, .5, .6, .7):
            data = {
                "censoring": [],
                "Harrel's C": [],
                "Uno's C": []
            }
            for _ in range(n_repeats):
                # generate data
                X_test, y_test, y_train, actual_c = self.generate_survival_data(cens)

                # estimate c-index
                c_harrell = concordance_index_censored(y_test["event"], y_test["time"], X_test)
                c_uno = concordance_index_ipcw(y_train, y_test, X_test)

                # save results
                data["censoring"].append(100. - y_test["event"].sum() * 100. / y_test.shape[0])
                data["Harrel's C"].append(actual_c - c_harrell[0])
                data["Uno's C"].append(actual_c - c_uno[0])

            # aggregate results
            for key, values in data.items():
                data_mean[key].append(np.mean(data[key]))
                data_std[key].append(np.std(data[key], ddof=1))

        data_mean = pd.DataFrame.from_dict(data_mean)
        data_std = pd.DataFrame.from_dict(data_std)
        return data_mean, data_std

    @staticmethod
    def plot_results(data_mean, data_std, **kwargs):
        index = pd.Index(data_mean["censoring"].round(3), name="mean percentage censoring")
        for df in (data_mean, data_std):
            df.drop("censoring", axis=1, inplace=True)
            df.index = index

        ax = data_mean.plot.bar(yerr=data_std, **kwargs)
        ax.set_ylabel("Actual C - Estimated C")
        ax.yaxis.grid(True)
        ax.axhline(0.0, color="gray")
