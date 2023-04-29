import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()
ROOT_DIRECTORY = os.environ["ROOT_DIRECTORY"]

pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv(ROOT_DIRECTORY + "/data/raw/games.csv")


def basic_features(dta):
    onlyr = dta.loc[dta["Rnd"].str.contains("R")]
    onlyr["game_year"] = pd.DatetimeIndex(onlyr["Date"]).year
    onlyr["round_num"] = onlyr["Rnd"].str.replace("R", "").astype("int")
    onlyr["win"] = 0
    # onlyr['R'] = onlyr['R'].fillna(0)
    onlyr.loc[onlyr["R"] == "W", "win"] = 1
    onlyr["winsum"] = onlyr["win"]
    onlyr["game_points"] = 0
    onlyr.loc[onlyr["R"] == "W", "game_points"] = 4
    onlyr.loc[onlyr["R"] == "D", "game_points"] = 2
    onlyr["home_win"] = 0
    onlyr.loc[(onlyr["R"] == "W") & (onlyr["T"] == "H"), "home_win"] = 1
    onlyr["home_for"] = 0
    onlyr.loc[(onlyr["T"] == "H"), "home_for"] = onlyr["F"]
    onlyr["home_against"] = 0
    onlyr.loc[(onlyr["T"] == "H"), "home_against"] = onlyr["A"]
    onlyr["home_game"] = 0
    onlyr.loc[(onlyr["T"] == "H"), "home_game"] = 1
    onlyr["away_win"] = 0
    onlyr.loc[(onlyr["R"] == "W") & (onlyr["T"] == "A"), "away_win"] = 1
    onlyr["away_game"] = 0
    onlyr["games"] = 1
    onlyr.loc[(onlyr["T"] == "A"), "away_game"] = 1
    onlyr["early"] = 0
    onlyr.loc[onlyr["round_num"] <= 5, "early"] = 1
    onlyr["mid"] = 0
    onlyr.loc[((onlyr["round_num"] >= 6) & (onlyr["round_num"] <= 11)), "mid"] = 1
    onlyr.loc[onlyr["round_num"] <= 5, "season_status"] = "early"
    onlyr.loc[
        (onlyr["round_num"] > 5) & (onlyr["round_num"] <= 12), "season_status"
    ] = "mid"
    onlyr.loc[(onlyr["round_num"] > 12), "season_status"] = "late"
    onlyr.sort_values(
        by=["Team", "game_year", "round_num"],
        ascending=[True, True, True],
        inplace=True,
    )
    onlyr["rolling_wins"] = (
        onlyr.groupby(["Team", "game_year"])["M"].cumsum() - onlyr["M"]
    )
    onlyr["rolling_game_points"] = (
        onlyr.groupby(["Team", "game_year"])["game_points"].cumsum()
        - onlyr["game_points"]
    )
    onlyr["rolling_for"] = (
        onlyr.shift(periods=1).groupby(["Team", "game_year"])["F"].cumsum() + 1
    ) - onlyr.shift(periods=1).groupby(["Team", "game_year"])["F"].cummin()
    -onlyr.shift(periods=1).groupby(["Team", "game_year"])["F"].cummax()
    onlyr["rolling_against"] = (
        onlyr.shift(periods=1).groupby(["Team", "game_year"])["A"].cumsum() + 1
    ) - onlyr.shift(periods=1).groupby(["Team", "game_year"])["A"].cummin()
    -onlyr.shift(periods=1).groupby(["Team", "game_year"])["A"].cummax()
    # onlyr["rolling_against"] = (
    #     onlyr.groupby(["Team", "game_year"])["A"].cumsum() + 1
    # ) - onlyr["A"]
    onlyr["home_rolling_for"] = (
        onlyr.shift(periods=1).groupby(["Team", "game_year"])["home_for"].cumsum() + 1
    ) - onlyr.shift(periods=1).groupby(["Team", "game_year"])["home_for"].cummin()
    -onlyr.shift(periods=1).groupby(["Team", "game_year"])["home_for"].cummax()
    onlyr["rolling_against"] = (
        onlyr.groupby(["Team", "game_year"])["A"].cumsum() + 1
    ) - onlyr["A"]
    onlyr["home_rolling_against"] = (
        onlyr.groupby(["Team", "game_year"])["home_against"].cumsum() + 1
    ) - onlyr["home_against"]
    onlyr["rolling_margin"] = onlyr["rolling_for"] - onlyr["rolling_against"]
    onlyr["rolling_percentage_1"] = (onlyr["F"].rolling(7).sum() - onlyr["F"]) / (
        onlyr["A"].rolling(7).sum() - onlyr["A"]
    )
    onlyr["rolling_wins_1"] = onlyr["game_points"].shift(1).rolling(3).sum()
    onlyr["rolling_wins_2"] = (
        onlyr["game_points"].shift(1).rolling(6).sum() - onlyr["rolling_wins_1"]
    )
    onlyr["rolling_wins_3"] = (
        onlyr["game_points"].shift(1).rolling(9).sum()
        - onlyr["rolling_wins_1"]
        - onlyr["rolling_wins_2"]
    )
    # onlyr["rolling_wins_1"] = (
    #     onlyr.shift(periods=1).groupby(["Team"])["M"].rolling(4).sum()
    # )
    # onlyr["rolling_F_1"] = onlyr["F"].rolling(7).sum() - onlyr["F"]
    # onlyr["rolling_A_1"] = onlyr["A"].rolling(7).sum() - onlyr["A"]
    onlyr["rolling_F_median_1"] = (
        onlyr["F"].shift(periods=1).rolling(10).median(method="lower")
    )
    onlyr["rolling_A_median_1"] = (
        onlyr["A"].shift(periods=1).rolling(10).median(method="lower")
    )
    onlyr["rolling_M_median_1"] = (
        onlyr["M"].shift(periods=1).rolling(10).median(method="lower")
    )
    onlyr["rolling_M_median_2"] = (
        onlyr["M"].shift(periods=7).rolling(5).median(method="lower")
    )
    onlyr["rolling_M_median_3"] = (
        onlyr["M"].shift(periods=11).rolling(7).median(method="lower")
    )
    onlyr["prev_match_M_1"] = onlyr["M"].shift(periods=1).rolling(5).sum()
    onlyr["prev_match_M_2"] = onlyr["M"].shift(periods=2)
    onlyr["prev_match_M_3"] = onlyr["M"].shift(periods=3)
    onlyr["prev_match_M_4"] = onlyr["M"].shift(periods=4)
    onlyr["prev_match_M_5"] = onlyr["M"].shift(periods=5)
    onlyr["prev_match_M_6"] = onlyr["M"].shift(periods=6)
    onlyr["prev_match_M_7"] = onlyr["M"].shift(periods=7)
    onlyr["prev_match_M_8"] = onlyr["M"].shift(periods=8)
    onlyr["prev_match_M_9"] = onlyr["M"].shift(periods=9)
    onlyr["prev_match_M_10"] = onlyr["M"].shift(periods=10)
    onlyr["prev_match_M_11"] = onlyr["M"].shift(periods=11)
    onlyr["prev_match_M_1"] = onlyr["win"].shift(periods=1)
    onlyr["prev_match_M_2"] = onlyr["win"].shift(periods=2)
    onlyr["prev_match_M_3"] = onlyr["win"].shift(periods=3)
    onlyr["prev_match_M_4"] = onlyr["win"].shift(periods=4)
    onlyr["prev_match_M_5"] = onlyr["win"].shift(periods=5)
    onlyr["prev_match_M_6"] = onlyr["win"].shift(periods=6)
    onlyr["prev_match_M_7"] = onlyr["win"].shift(periods=7)
    onlyr["prev_match_M_8"] = onlyr["win"].shift(periods=8)
    onlyr["prev_match_M_9"] = onlyr["win"].shift(periods=9)
    onlyr["prev_match_M_10"] = onlyr["win"].shift(periods=10)
    onlyr["prev_match_M_11"] = onlyr["win"].shift(periods=11)
    onlyr["rolling_M_3"] = onlyr["M"].rolling(16).std() - onlyr["win"]
    onlyr["rolling_games"] = onlyr.groupby(["Team", "game_year"])["win"].cumcount() + 1
    onlyr["rolling_percentage"] = onlyr["rolling_for"] / onlyr["rolling_against"]
    # onlyr['rmc_c'] = onlyr['rmc_c'].fillna(0)
    onlyr["game_key"] = (
        onlyr["Crowd"].astype("str")
        + onlyr["Date"].astype("str")
        + onlyr["round_num"].astype("str")
    )
    onlyr = onlyr.fillna(0)
    return onlyr


def ladder_merge(dta):
    aggregation = {
        "game_points": "sum",
        "F": "sum",
        "A": "sum",
        "home_game": "sum",
        "home_win": "sum",
        "away_game": "sum",
        "away_win": "sum",
        "winsum": "sum",
        "games": "sum",
    }
    ladder = dta.groupby(["Team", "game_year"]).agg(aggregation).reset_index()
    ladder["percentage"] = ladder["F"] / ladder["A"]
    ladder["percentage_wins"] = ladder["winsum"] / ladder["games"]
    ladder["home_percentage_wins"] = ladder["home_win"] / ladder["home_game"]
    ladder["away_percentage_wins"] = ladder["away_win"] / ladder["away_game"]
    ladder.sort_values(
        by=["game_year", "game_points", "percentage"],
        ascending=[False, False, False],
        inplace=True,
    )
    ladder["ladder_position"] = ladder.groupby("game_year").cumcount() + 1
    ladder["join_year"] = ladder["game_year"] + 1
    ladder.drop(columns="game_year", inplace=True)
    col_list = [
        "Rnd",
        "R",
        "Team",
        "Opponent",
        "Date",
        "game_year",
        "game_key",
        "round_num",
        "T",
        "win",
        "season_status",
        "rolling_for",
        "M",
        "rolling_wins",
        "rolling_games",
        "rolling_against",
        "rolling_margin",
        "rolling_percentage",
        # "rolling_F_1",
        # "rolling_A_1",
        "rolling_F_median_1",
        "rolling_A_median_1",
        "rolling_wins_1",
        "rolling_wins_2",
        "rolling_wins_3",
        "prev_match_M_1",
        "rolling_M_median_1",
        "rolling_M_median_2",
        "rolling_M_median_3",
        "prev_match_M_2",
        "prev_match_M_3",
        "prev_match_M_4",
        "prev_match_M_5",
        "prev_match_M_6",
        "prev_match_M_7",
        "prev_match_M_8",
        "prev_match_M_9",
        "prev_match_M_10",
        "prev_match_M_11",
        "rolling_game_points",
        "home_rolling_for",
        "home_rolling_against",
        "rolling_percentage_1",
        "home_for",
        "home_against",
        "early",
        "mid",
    ]
    feat = dta[col_list]
    lad_merged = pd.merge(
        left=feat,
        right=ladder,
        left_on=["Team", "game_year"],
        right_on=["Team", "join_year"],
    )
    merged = pd.merge(
        left=lad_merged,
        right=lad_merged,
        left_on=["Opponent", "Date", "round_num"],
        right_on=["Team", "Date", "round_num"],
        suffixes=("", "_joined"),
    )
    merged.sort_values(
        by=["Team", "game_year", "round_num"],
        ascending=[True, True, True],
        inplace=True,
    )
    merged = merged.loc[merged["T"] == "H"]
    merged["game_year"] = pd.DatetimeIndex(merged["Date"]).year
    xd = merged.groupby(["Team"])[["game_year"]].min().reset_index()
    xd["rem_flag"] = 1
    xy = pd.merge(
        left=merged,
        right=xd,
        on=["Team", "game_year"],
        how="left",
        suffixes=["", "rem_"],
    )
    final_output = xy.loc[xy["rem_flag"].isna()]
    return final_output


output = ladder_merge(basic_features(df))
# print(output.head(5))

output.to_csv(ROOT_DIRECTORY + "/data/processed/processed_features.csv")
