from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt


def score(clf, x, y):
    return roc_auc_score(y == 1, clf.predict_proba(x)[:, 1])


class Boosting:
    def __init__(
            self,
            base_model_class=DecisionTreeRegressor,
            base_model_params: dict = None,
            n_estimators: int = 10,
            learning_rate: float = 0.1,
            subsample: float = 0.1,
            early_stopping_rounds: int = None,
            plot: bool = False,
    ):
        self.base_model_class = base_model_class
        self.base_model_params = {} if base_model_params is None else base_model_params
        self.n_estimators = n_estimators
        self.models = []
        self.gammas = []
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.early_stopping_rounds = early_stopping_rounds
        self.plot = plot
        self.history = defaultdict(list)
        self.sigmoid = lambda x: 1 / (1 + np.exp(-x))
        self.loss_fn = lambda y, z: -np.log(self.sigmoid(y * z)).mean()
        self.loss_derivative = lambda y, z: -y * self.sigmoid(-y * z)

        if early_stopping_rounds is not None:
            self.validation_loss = np.full(self.early_stopping_rounds, np.inf)

    def fit_new_base_model(self, x, y, predictions):
        # Generate a bootstrap sample
        bootstrap_indices = np.random.choice(np.arange(x.shape[0]), size=int(self.subsample * x.shape[0]), replace=True)
        x_bootstrap = x[bootstrap_indices]
        y_bootstrap = y[bootstrap_indices]

        # Fit the base model
        model = self.base_model_class(**self.base_model_params)
        residuals = y_bootstrap - self.sigmoid(predictions[bootstrap_indices])
        model.fit(x_bootstrap, residuals)
        new_predictions = model.predict(x)

        # Optimize gamma
        gamma = self.find_optimal_gamma(y, predictions, new_predictions)
        self.gammas.append(gamma)
        self.models.append(model)

        return gamma, new_predictions

    def fit(self, x_train, y_train, x_valid, y_valid):
        train_predictions = np.zeros(y_train.shape[0])
        valid_predictions = np.zeros(y_valid.shape[0])
        best_score = -np.inf
        no_improvement_count = 0

        for i in range(self.n_estimators):
            gamma, new_train_predictions = self.fit_new_base_model(x_train, y_train, train_predictions)
            train_predictions += self.learning_rate * gamma * new_train_predictions
            valid_predictions += self.learning_rate * gamma * self.models[-1].predict(x_valid)

            train_loss = self.loss_fn(y_train, train_predictions)
            valid_loss = self.loss_fn(y_valid, valid_predictions)
            self.history['train_loss'].append(train_loss)
            self.history['valid_loss'].append(valid_loss)

            if self.early_stopping_rounds is not None:
                if valid_loss < np.min(self.validation_loss):
                    self.validation_loss = np.roll(self.validation_loss, shift=-1)
                    self.validation_loss[-1] = valid_loss
                    best_score = valid_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= self.early_stopping_rounds:
                        print(f"Early stopping after {i+1} estimators")
                        break

        if self.plot:
            plt.figure(figsize=(10, 5))
            plt.plot(self.history['train_loss'], label='Train Loss')
            plt.plot(self.history['valid_loss'], label='Valid Loss')
            plt.xlabel('Number of Estimators')
            plt.ylabel('Loss')
            plt.legend()
            plt.title('Boosting Training and Validation Loss')
            plt.show()

    def predict_proba(self, x):
        predictions = np.zeros(x.shape[0])
        for gamma, model in zip(self.gammas, self.models):
            predictions += self.learning_rate * gamma * model.predict(x)
        return np.vstack((1 - self.sigmoid(predictions), self.sigmoid(predictions))).T

    def find_optimal_gamma(self, y, old_predictions, new_predictions):
        gammas = np.linspace(start=0, stop=1, num=100)
        losses = [self.loss_fn(y, old_predictions + gamma * new_predictions) for gamma in gammas]
        return gammas[np.argmin(losses)]

    def score(self, x, y):
        return score(self, x, y)

    @property
    def feature_importances_(self):
        total_importances = np.zeros(self.models[0].feature_importances_.shape)
        for gamma, model in zip(self.gammas, self.models):
            total_importances += gamma * model.feature_importances_
        return total_importances / sum(self.gammas)

   
