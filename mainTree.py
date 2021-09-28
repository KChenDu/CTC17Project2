from utils import *
from tree import DecisionTree

data, results, possible_results, factors, factors_values = get_data("accident_data.csv")

limiter = len(data) * 4 // 5

training_data = data[:limiter]
testing_data = data[limiter:]

training_results = results[:limiter]
testing_results = results[limiter:]

tree = DecisionTree(training_data, training_results, possible_results, factors, factors_values)
tree.show()
