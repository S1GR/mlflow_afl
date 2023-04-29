import pandas as pd
import yaml
from flaml import AutoML
import pickle

from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def read_yaml_config():
    with open("config.yaml", "r") as stream:
        config_input = yaml.safe_load(stream)
    return config_input


def read_and_prepare_dataset(feature_set, test_year):
    df = pd.read_csv(feature_set)
    df["Date"] = pd.to_datetime(df["Date"])
    df_test = df.loc[df["Date"].dt.year == test_year]
    df_train = df.loc[df["Date"].dt.year != test_year]
    return df_test, df_train


@ignore_warnings(category=ConvergenceWarning)
def train_model(training_set, features, target, model_settings):
    automl = AutoML()
    automl.fit(training_set[features], training_set[target], **model_settings)
    print("Best ML leaner:", automl.best_estimator)
    print("Best hyperparmeter config:", automl.best_config)
    print("Best log_loss on validation data: {0:.4g}".format(1 - automl.best_loss))
    print(
        "Training duration of best run: {0:.4g} s".format(automl.best_config_train_time)
    )
    return automl


def save_model(model, loss):
    with open("models/automl" + str(loss) + ".pkl", "wb") as f:
        pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)
