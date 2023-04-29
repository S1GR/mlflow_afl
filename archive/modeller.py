from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import make_scorer
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier


class compressor(BaseEstimator, ClassifierMixin):
    # initializer
    def __init__(self, adj):
        # save the features list internally in the class
        self.adj = adj

    def fit(self, X, y=None):
        return self

    def predict_proba(self, X, y=None):
        # return the dataframe with the specified features
        # return pd.DataFrame(X)
        dfr = pd.DataFrame(X)
        dfr["newer"] = dfr[0] - ((dfr[0] - 0.5) * self.adj)
        dfr["a"] = 1 - dfr["newer"]
        return dfr[["a", "newer"]].to_numpy()  # .reshape(-1,1)


def new_scorer2(y, y_pred):
    xx = pd.DataFrame({"y": list(y), "y_pred": list(y_pred)}, columns=["y", "y_pred"])
    xx.loc[xx["y"].astype(int) == 0, "actual_score"] = 1 + np.log2(
        1 - xx["y_pred"].astype(float)
    )
    xx.loc[xx["y"].astype(int) == 1, "actual_score"] = 1 + np.log2(
        xx["y_pred"].astype(float)
    )
    return np.mean(xx["actual_score"])


# Utility function to report best scores
def report2(results, n_top=10):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")


def create_model2(model, categorical_columns, numerical_columns):
    categorical_encoder = OneHotEncoder(handle_unknown="ignore", drop="first")
    # categorical_encoder = TargetEncoder()
    numerical_pipe = Pipeline([("imputer", SimpleImputer(strategy="mean"))])
    preprocessing = ColumnTransformer(
        [
            ("cat", categorical_encoder, categorical_columns),
            ("num", numerical_pipe, numerical_columns),
        ]
    )

    estimators = [("inp_model", model)]

    clf = StackingClassifier(estimators=estimators, final_estimator=compressor(0.1))

    lg = Pipeline(
        [
            ("preprocess", preprocessing),
            ("scl", StandardScaler()),
            ("classifier", clf),
        ]
    )
    # lg.fit(training_set[features],training_set[target])
    return lg


def searcher(param_grid, model, searcher):
    n_iter_search = searcher
    afl_scorer = make_scorer(new_scorer2, needs_proba=True, greater_is_better=True)
    shuffler = ShuffleSplit(n_splits=30, test_size=0.2, random_state=1)
    random_search = RandomizedSearchCV(
        model,
        param_distributions=param_grid,
        n_iter=n_iter_search,
        scoring=afl_scorer,
        refit=True,
        cv=shuffler,
    )
    return random_search
