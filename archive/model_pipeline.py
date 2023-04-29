import pandas as pd
from archive.helpers import read_yaml_config, read_and_prepare_dataset, train_model

# from warnings import simplefilter
# from sklearn.exceptions import ConvergenceWarning
# simplefilter("ignore", category=ConvergenceWarning)


# Read YAML file
config_input = read_yaml_config()

# FLAML
FLAML_SETTINGS = config_input["flaml"]["settings"]

# Data
FEATURE_SET = config_input["data"]["feature_set"]
TEST_YEAR = config_input["data"]["training"]["test_year"]
TARGET = config_input["data"]["training"]["target"]
TRAIN_COLUMNS = config_input["data"]["training"]["train_cols"]


df_test, df_train = read_and_prepare_dataset(FEATURE_SET, TEST_YEAR)

trained_model = train_model(df_train, TRAIN_COLUMNS, TARGET, FLAML_SETTINGS)
