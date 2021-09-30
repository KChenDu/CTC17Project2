from numpy import positive
from utils import *
from tree import DecisionTree
from priori import PrioriClassifier

data, results, possible_results, factors, factors_values = get_data("accident_data.csv")

limiter = len(data) * 4 // 5

training_data = data[:limiter]
testing_data = data[limiter:]

training_results = results[:limiter]
testing_results = results[limiter:]

# Tree
tree = DecisionTree(training_data, training_results, possible_results, factors, factors_values)
tree_prediction = tree.predict(testing_data)

# Priori
labelToInt = {
    'I':   1,
    'II':  2,
    'III': 3,
    'IV':  4,
    'V':   5
}
priori = PrioriClassifier(training_results, labelToInt)
mode_prediction = priori.predict(testing_data, strategy='mode')
mean_prediction = priori.predict(testing_data, strategy='truncatedMean', fraction=0.1)
