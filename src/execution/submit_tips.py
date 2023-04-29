from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas as pd
from dotenv import load_dotenv
import os

load_dotenv()

TIP_SITE_USERNAME = os.environ["TIP_SITE_USERNAME"]
TIP_SITE_PASSWORD = os.environ["TIP_SITE_PASSWORD"]

browser = webdriver.Firefox()

browser.get("http://probabilistic-footy.monash.edu/~footy/tips.shtml")

df_mapped = pd.read_csv("data/scored/scored_2023_7_automl_xgb_4000.csv")

time.sleep(3)
username = browser.find_element("name", "name")
password = browser.find_element("name", "passwd")
username.send_keys(TIP_SITE_USERNAME)
password.send_keys(TIP_SITE_PASSWORD)
login_attempt = browser.find_element("xpath", "//*[@type='submit']")
login_attempt.submit()

time.sleep(3)
main_table = browser.find_elements(By.TAG_NAME, "tbody")
rower = main_table[1].find_elements(By.TAG_NAME, "tr")
for rows in range(len(rower) - 1):
    home_team = rower[rows + 1].find_elements(By.TAG_NAME, "td")[2].text
    try:
        prediction = df_mapped.loc[df_mapped["Submit_team"] == home_team][
            "score"
        ].values[0]
        gamer = browser.find_element("name", "game" + str(rows + 1))
        gamer.clear()
        gamer.send_keys(prediction.astype("str"))
    except:
        print("Game not found")

time.sleep(3)
login_attempt = browser.find_element("xpath", "//*[@type='submit']")
# login_attempt.submit()
