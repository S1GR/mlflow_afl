# Summary

The purpose of this repo is to build a basic model to predict the likelihood that the home team will win a game of AFL.

Includes the following key components:
* Model training
* Model scoring
* Capability to enable submitting tips to Monash probabilistic tipping site

Note that this repo doesn't yet integrate MLFlow so apologies to anyone reading this for the misleading title ; )

## Model training

Instructions to be updated.

## Model scoring

Assumes that you have a trained model from above step.

Open "pull_data" and change the round accordingly

Run 
~~~
python /Users/scottgregory/Documents/gits/mlflow_afl/src/data/pull_data.py
~~~

Open "data_prep" and change the round accordingly

Run 
~~~
python /Users/scottgregory/Documents/gits/mlflow_afl/src/features/data_prep.py
~~~

Open "score_model" and change the round accordingly

Run
~~~
python /Users/scottgregory/Documents/gits/mlflow_afl/src/models/score_model.py
~~~

Open "submit_tips" and change the round accordingly

Run below from parent src/execution directory
~~~
python submit_tips.py
~~~

TODO

* Add the above steps to a script with parameters for round number so there's a single execution step
* Investigate MLFlow integration.  Need a sustainable way to be able to experiment and keep visibility of best feature/model/hyperparameter combinations
* Further profiling.  Interested to understand whether home game advantage is consistent across seasons
* Update the pull data script so that it doesn't need to do a full refresh every time (i.e. delta only)