# IKT440-Project

This repo contains code developed in the ICT440 project in the fall of 2019. The assignment is to compare different machine learning algorithms on the connect-4 data set and create an AI player for the game.

## Getting started
In order to run the game, MCTS.py, the machine learning model must first be trained and exportet to a file. Run the file Run the "export_rf_model.py" script to save the trained random forest model to file, before running MCTS.py


## Contents of the repository

This repository contains the scripts used to generate results for the report, a script to tune hyper-parameters of the different algorthms and the connect-4 game with an AI oponent using a random forest model trained on the connect 4 data set.

### Algorithms
The algorithms folder contains scripts to train the baseline and optimized models in order to generate the results shown in the report.

### Tune.py
The script _Tune.py_ is used to perform hyper-parameter tuning on the different algorithms using _GridSearchCV_. The script is set up with hyper parameters for the different algorithms. Simply swap out the the _estimator_ and _params_grid_ parameters to tune the different algorithms.

### MCTS.py


