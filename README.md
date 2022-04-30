# KNN-Vehicle-Evaluation

## Credit: https://github.com/techwithtim

A repository containing a vehicle evaluation model utilizing the K-Nearest Neighbors algorithm.

Language: Python  
Libraries: pandas, numpy, sklearn  
Dataset: car.data from the UCI (University of California, Irvine) machine learning repository

## Intro

The model takes the overall buying price, maintenance cost, number of doors, capacity of number of persons, size of the luggage boot, and safety level of a specific vehicle, and attempts to predict the vehicle's "acceptability"

Acceptability (cls column in car.data) is determined by "... a simple hierarchical decision model originally developed for the demonstration of DEX, M. Bohanec, V. Rajkovic: Expert system for decision making. (Sistemica 1(1), pp. 145-157, 1990.)."

Dataset was collected from the UCI machine learning repository, an archive containing 557 datasets (https://archive.ics.uci.edu/ml/datasets/Car+Evaluation)

## /

**KNN.py:**  
The main algorithm (See section: "Design description" below)

**car.data:**  
A csv table containing vehicle evaluation information, with rows being each vehicle design, and columns being features such as: overall buying price, maintenance cost, number of doors, etc.

## Design description

1) Retrieves dataset of vehicle designs, and transforms columns with string classification values into integers utilizing sklearn's preprocessing library

2) The resulting matrix is of type panda data frame, so we pull each column, converting them into a numpy array, as sklearn's modelling functions only accept arrays

3) Combines converted columns into one matrix, and dataset is divided by 90% for training purposes, and 10% for testing purposes

4) The model is fit utilizing the K-Nearest Neighbors algorithm, with k = 9 (Number of neighbors was increased/decreased arbitrarily until high, acceptable accuracy was achieved)

The model does not need to be saved, as due to the characteristics of KNN models, every time it needs to make a prediction, a new training point is added, so it must traverse and calculate all distances between each point again

## Outcome

Test accuracy achieved: 0.9422
