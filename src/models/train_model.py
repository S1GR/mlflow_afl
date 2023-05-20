import pandas as pd
import pickle
import pandas as pd
from flaml import AutoML
import yaml
from dotenv import load_dotenv
import os
from custom_metric import custom_metric

load_dotenv()

ROOT_DIRECTORY = os.environ["ROOT_DIRECTORY"]

# Read YAML file
with open(ROOT_DIRECTORY + "/src/config.yaml", "r") as stream:
    config_input = yaml.safe_load(stream)

# FLAML
FLAML_SETTINGS = config_input["flaml"]["settings"]

# Data
FEATURE_SET = config_input["data"]["feature_set"]
TEST_YEAR = config_input["data"]["training"]["test_year"]
TARGET = config_input["data"]["training"]["target"]
TRAIN_COLUMNS = config_input["data"]["training"]["train_cols"]
SETTINGS = config_input["flaml"]["settings"]


df = pd.read_csv(ROOT_DIRECTORY + FEATURE_SET)
df = df.loc[df["game_year"] >= 1995]
# df["Date"] = pd.to_datetime(df["Date"])
# df = df.loc[df['round_num']<=10]
df_test = df.loc[df["game_year"] == TEST_YEAR]
df_train = df.loc[df["game_year"] < TEST_YEAR]

print("Training model")
automl = AutoML()
automl.fit(df_train[TRAIN_COLUMNS], df_train[TARGET], metric=custom_metric, **SETTINGS)

print("Best ML leaner:", automl.best_estimator)
print("Best hyperparmeter config:", automl.best_config)
print("Best mean game score: {0:.4g}".format(-automl.best_loss))
print("Training duration of best run: {0:.4g} s".format(automl.best_config_train_time))

print("Writing model to file")
with open(ROOT_DIRECTORY + "/models/automl_lgbm_30_nf_state.pkl", "wb") as f:
    pickle.dump(automl, f, pickle.HIGHEST_PROTOCOL)
