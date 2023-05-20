from models.score_model import score_model
from features.data_prep import process_features
from data.pull_data import pull_raw_data
from execution.submit_tips import submit_tips

round_num = 10

pull_raw_data(round_num)
process_features()
score_model(round_num)
submit_tips(round_num)
