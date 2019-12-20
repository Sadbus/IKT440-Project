# IKT440-Project

This repository contains code developed in the ICT440 project in the fall of 2019. The assignment was to compare different machine learning algorithms on the Connect-4 data set and create an AI player for the game.

## Getting started
In order to run the game, MCTS.py, the machine learning model must first be trained and exported to a file. Run the "export_rf_model.py" script to save the trained random forest model to file, before running MCTS.py. The model to laod can be changed on line 31 in MCTS.py.

Please note that saving the SVM model to files takes approximatley 80 minutes. _probability_ is set to _true_ in order for MCTS.py to use _predict_proba()_. Thus, the fit method performs an extra 5 crossfolds on the training data.
## Contents of the repository

This repository contains the scripts used to generate results for the report, a script to tune hyper-parameters of the different algorithms and the Connect Four game with an AI opponent using a random forest model trained on the Connect-4 data set.

### Algorithms
The algorithms folder contains scripts to train the baseline and optimized models in order to generate the results shown in the report.

### Tune.py
The script _Tune.py_ is used to perform hyper-parameter tuning on the different algorithms using _GridSearchCV_. The script is set up with hyper parameters for the different algorithms. Simply swap out the the _estimator_ and _params_grid_ parameters to tune the different algorithms.

### MCTS.py

The Monte Carlo Tree Search AI player. To disable the Random Forest hybrid, set _use_random_forest_ to 0. If you want to have the AI play itself set _human_player_ to 0.
