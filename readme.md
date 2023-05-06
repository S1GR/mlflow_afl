# Summary

The purpose of this repo is to build a basic model to predict the likelihood that the home team will win a game of AFL.

Includes the following key components:
* Model training
* Model scoring
* Capability to enable submitting tips to Monash probabilistic tipping site.  If you end up using this repo to enter tips for this competition it would be great if you could try to modify in order to ensure that we're not entering the exact same tips
* Shap explanations  

Note that this repo doesn't yet integrate MLFlow so apologies to anyone reading this for the misleading title ; )

### Prerequisites

Must include a file called .env in the root of the project with the following fields populated.  The first is the location of the project on your local file system.  This is to help ensure that this can easily be run on different machines.

The second two items contain your username and password for the Monash tipping competition mentioned above.  You only need to fill these in if you wish to use the "submit_tips.py" module to submit tips automatically.

ROOT_DIRECTORY = "/Users/scottgregory/Documents/gits/mlflow_afl"
TIP_SITE_USERNAME = "xxx"
TIP_SITE_PASSWORD = "xxx"

## Model training

Instructions to be updated.

## Model scoring

Assumes that you have a trained model from above step.

Open src/execute.py
Update the round to reflect the round you intend to score (note that some settings such as year and model name are fixed but can be manually updated if required)
run
~~~
execute.py
~~~

TODO

* Currently the "custom_metric.py" is repeated in three places.  Need to figure out a way to only need this once
* Investigate MLFlow integration.  Need a sustainable way to be able to experiment and keep visibility of best feature/model/hyperparameter combinations
* Further profiling.  
 * Interested to understand whether home game advantage is consistent across seasons
 * Should there be an interaction effect between features and stage of the season.  Intuitively the model should care less about the last 7 games in round 1 than it would in round 20 but so far the model doesn't behave in that way
* Update the pull data script so that it doesn't need to do a full refresh every time (i.e. delta only)