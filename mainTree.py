from utils import *

data, results, factors = get_data("accident_data.csv")

limiter = len(data) * 4 // 5

training_data = data[:limiter]
testing_data = data[limiter:]

training_results = results[:limiter]
testing_results = results[limiter:]

