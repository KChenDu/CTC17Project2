import csv
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import metrics


def get_data(file):
    num_columns = 6
    data = []
    results = []
    possible_results = []
    factors = []
    factors_values = [[], [], [], [], [], []]
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if reader.line_num < 2:
                factors = [row[2], row[3], row[5], row[6], row[7], row[8]]
            else:
                data.append([row[2], row[3], row[5], row[6], row[7], row[8]])
                results.append(row[4])
                if not row[4] in possible_results:
                    possible_results.append(row[4])
                for i in range(num_columns):
                    if not data[-1][i] in factors_values[i]:
                        factors_values[i].append(data[-1][i])
    return data, results, possible_results, factors, factors_values

def plotConfusionMatrix(true_results, prediction, labels=None, title=None):
    cf_matrix = metrics.confusion_matrix(true_results, prediction)
    sns.heatmap(cf_matrix/np.sum(cf_matrix), fmt='.1%', annot=True,
        xticklabels=labels, yticklabels=labels, cmap='Blues', center=0.3,
        vmin=0.0, vmax=0.75)

    plt.title(title)
    plt.ylabel('True Accident level')
    plt.xlabel('Predicted Accident level')
    plt.savefig(title + '.png')
    plt.show()
