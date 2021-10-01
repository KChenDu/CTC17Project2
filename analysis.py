from numpy import positive
from utils import *
from tree import DecisionTree
from priori import PrioriClassifier
from tabulate import tabulate, tabulate_formats

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
priori_prediction = mean_prediction

# Analysis
int_testing_results = [labelToInt[label] for label in testing_results]
int_tree_prediction = [labelToInt[label] for label in tree_prediction]
int_priori_prediction = [labelToInt[label] for label in priori_prediction]

headers = ['Metric', 'Decision Tree', 'A Priori Classifier']

hitRate = ['Hit Rate', metrics.accuracy_score(int_testing_results, int_tree_prediction), 
    metrics.accuracy_score(int_testing_results, int_priori_prediction)]
meanSquaredError = ['Mean Squared Error', metrics.mean_squared_error(int_testing_results, int_tree_prediction),
    metrics.mean_squared_error(int_testing_results, int_priori_prediction)]
kappa = ['Kappa', metrics.cohen_kappa_score(int_testing_results, int_tree_prediction),
    metrics.cohen_kappa_score(int_testing_results, int_priori_prediction)] 

print(tabulate([hitRate, meanSquaredError, kappa], headers=headers))

plotConfusionMatrix(int_testing_results, int_tree_prediction, labelToInt.keys(), "Decision Tree Confusion Matrix")

plotConfusionMatrix(int_testing_results, int_priori_prediction, labelToInt.keys(), "A Priori Classifier Confusion Matrix")
