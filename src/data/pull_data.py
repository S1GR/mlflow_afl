from dotenv import load_dotenv
import os

from extract_helpers import get_teams, consolidate_table

load_dotenv()
ROOT_DIRECTORY = os.environ["ROOT_DIRECTORY"]

rounder = 7
df = consolidate_table(get_teams(), 40, rounder)
df.to_csv(ROOT_DIRECTORY + "/data/raw/games.csv")
