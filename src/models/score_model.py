import pandas as pd
import pickle
import pandas as pd
from flaml import AutoML
import yaml
from dotenv import load_dotenv
import os


load_dotenv()

ROOT_DIRECTORY = os.environ["ROOT_DIRECTORY"]
SCORE_YEAR = 2023
MODEL = "automl_lgbm_30_nf_state"

with open(ROOT_DIRECTORY + "/src/config.yaml", "r") as stream:
    config_input = yaml.safe_load(stream)

FEATURE_SET = config_input["data"]["feature_set"]
TRAIN_COLUMNS = config_input["data"]["training"]["train_cols"]


with open(ROOT_DIRECTORY + "/models/" + MODEL + ".pkl", "rb") as f:
    automl = pickle.load(f)


def score_model(score_round):
    print("Reading features.")
    df = pd.read_csv(ROOT_DIRECTORY + FEATURE_SET)
    current_round = df.loc[
        (df["game_year"] == SCORE_YEAR) & (df["round_num"] == score_round)
    ]

    # print(current_round.head(1))
    print("Scoring model.")
    current_round["score"] = automl.predict_proba(current_round[TRAIN_COLUMNS])[:, 1]

    mapper = pd.read_csv(ROOT_DIRECTORY + "/data/mapping_tables/submit_mapper.csv")
    df_mapped = pd.merge(current_round, mapper, left_on="Team", right_on="Season_team")[
        ["Submit_team", "Opponent", "score"] + TRAIN_COLUMNS
    ]

    df_mapped.to_csv(
        ROOT_DIRECTORY
        + "/data/scored/scored_"
        + str(SCORE_YEAR)
        + "_"
        + str(score_round)
        + "_"
        + str(MODEL)
        + ".csv",
        index=False,
    )
    print("Model scored.")


if __name__ == "__main__":
    score_model(7)
