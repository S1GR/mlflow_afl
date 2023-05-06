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


# def custom_metric(
#     X_val,
#     y_val,
#     estimator,
#     labels,
#     X_train,
#     y_train,
#     weight_val=None,
#     weight_train=None,
#     *args,
# ):
#     from sklearn.metrics import log_loss
#     import time

#     start = time.time()
#     y_pred = estimator.predict_proba(X_val)
#     pred_time = (time.time() - start) / len(X_val)
#     val_loss = log_loss(y_val, y_pred, labels=labels, sample_weight=weight_val)
#     y_pred = estimator.predict_proba(X_train)
#     train_loss = log_loss(y_train, y_pred, labels=labels, sample_weight=weight_train)
#     alpha = 0.5
#     return val_loss * (1 + alpha) - alpha * train_loss, {
#         "val_loss": val_loss,
#         "train_loss": train_loss,
#         "pred_time": pred_time,
#     }
