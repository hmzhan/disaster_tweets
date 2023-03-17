from sksurv.datasets import load_flchain
from sklearn.model_selection import train_test_split


class Data:
    def __init__(self):
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        x, y = load_flchain()
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=0)

    def impute_data(self):

