import pandas as pd
import numpy as np


def score_function(dataf):
    dataf.loc[dataf["target"] == 1, "prob_score"] = 1 + np.log2(dataf["score"])
    dataf.loc[dataf["target"] == 0, "prob_score"] = 1 + np.log2(1 - dataf["score"])
    per_game_score = dataf["prob_score"].sum() / len(dataf)
    return -per_game_score


def custom_metric(
    X_val,
    y_val,
    estimator,
    labels,
    X_train,
    y_train,
    weight_val=None,
    weight_train=None,
    *args,
):
    y_pred = estimator.predict_proba(X_val)[:, 1]
    val_df = pd.DataFrame({"score": y_pred, "target": y_val})
    metric_to_optimise = score_function(val_df)
    return metric_to_optimise, {
        "val_loss": 1,
        "train_loss": 1,
        "pred_time": 1,
    }
