import requests
from bs4 import BeautifulSoup
import pandas as pd
import datetime


def get_teams():
    page = requests.get("https://afltables.com/afl/seas/2023.html")
    soup = BeautifulSoup(page.content, "html.parser")
    tables = soup.find_all("td", width="85%")
    teams_list = []
    for i in range(3):
        teams_list.append(
            [
                (x.get_text(), x["href"].split("/")[2].split("_")[0])
                for x in tables[i].find_all("a")
                if "teams" in x["href"]
            ]
        )
    return list(set([x for team in teams_list for x in team]))


def table_reader(team_tuple, games):
    df_list = []
    tabler = pd.read_html(
        "https://afltables.com/afl/teams/" + team_tuple[1] + "/allgames.html"
    )
    num_games = len(tabler) if len(tabler) < games else games
    for i in range(num_games):
        single_season = tabler[i]
        single_season.columns = single_season.columns.droplevel(-1)
        single_season_filt = single_season.loc[
            ~single_season["Rnd"].isin(["Totals", "Averages"])
        ]
        single_season_filt["Team"] = team_tuple[0]
        single_season_filt.drop(columns="Scoring", inplace=True)
        single_season_filt["Date"] = pd.to_datetime(single_season_filt["Date"])
        df_list.append(single_season_filt)
    return pd.concat(df_list)


def consolidate_table(team_list, games, round_scorer=""):
    master_table_list = []
    for i in team_list:
        master_table_list.append(table_reader(i, games))
    x = pd.concat(master_table_list)
    if round_scorer:
        # print(round_scorer)
        rs = round_adder(round_scorer)
        # print(rs.dtypes)
        # print(x.dtypes)
        return pd.concat([x, rs])
    else:
        return x


def round_adder(round_num):
    tabler = pd.read_html("https://en.wikipedia.org/wiki/2023_AFL_season")
    tt = tabler[round_num + 3]
    print(tt)
    if (len(tt) == 10) | (len(tt) == 12):
        rounder = "R" + str(round_num)
        tt = tt.loc[tt[2] == "vs."]
        tt["Rnd"] = rounder
        tt["T"] = "H"
        tt["Opponent"] = tt[3]
        tt["F"] = 0
        tt["A"] = 0
        tt["R"] = "U"
        tt["M"] = 0
        tt["W-D-L"] = "U"
        tt["Venue"] = tt[4]
        tt["Crowd"] = 0
        tt["Date"] = datetime.datetime(2023, 12, 31)
        tt["Team"] = tt[1]
        cols = [
            "Rnd",
            "T",
            "Opponent",
            "F",
            "A",
            "R",
            "M",
            "W-D-L",
            "Venue",
            "Crowd",
            "Date",
            "Team",
        ]
        table_out = tt[cols]
        aa = table_out.copy()
        aa["opp_stage"] = aa["Opponent"]
        aa["Opponent"] = aa["Team"]
        aa["Team"] = aa["opp_stage"]
        aa["T"] = "A"
        aa = aa[cols]
        table_out = pd.concat([table_out, aa])
    return table_out


# def round_adder(round_num):
#     tabler = pd.read_html("https://en.wikipedia.org/wiki/2023_AFL_season")
#     for tt in tabler:
#         if (len(tt) == 10) | (len(tt) == 12):
#             if str(tt.iloc[[1]][2].values[0].split(" ")[1]) == str(round_num):
#                 rounder = "R" + (tt.iloc[[1]][2].values[0].split(" ")[1])
#                 tt = tt.loc[tt[2] == "vs."]
#                 tt["Rnd"] = rounder
#                 tt["T"] = "H"
#                 tt["Opponent"] = tt[3]
#                 tt["F"] = 0
#                 tt["A"] = 0
#                 tt["R"] = "U"
#                 tt["M"] = 0
#                 tt["W-D-L"] = "U"
#                 tt["Venue"] = tt[4]
#                 tt["Crowd"] = 0
#                 tt["Date"] = datetime.datetime(2023, 12, 31)
#                 tt["Team"] = tt[1]
#                 cols = [
#                     "Rnd",
#                     "T",
#                     "Opponent",
#                     "F",
#                     "A",
#                     "R",
#                     "M",
#                     "W-D-L",
#                     "Venue",
#                     "Crowd",
#                     "Date",
#                     "Team",
#                 ]
#                 table_out = tt[cols]
#                 aa = table_out.copy()
#                 aa["opp_stage"] = aa["Opponent"]
#                 aa["Opponent"] = aa["Team"]
#                 aa["Team"] = aa["opp_stage"]
#                 aa["T"] = "A"
#                 aa = aa[cols]
#                 table_out = pd.concat([table_out, aa])
#     return table_out
