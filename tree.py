from treelib import Tree
import statistics
import copy
import numpy as np


class DecisionTree:

    def __init__(self, data, results, possible_results, factors, factors_values):
        self.node_num = 0
        self.tree = Tree()
        self.possible_results = possible_results
        self.num_possible_results = len(possible_results)
        self.get_tree(copy.deepcopy(data), results[:], factors[:], copy.deepcopy(factors_values), statistics.mode(results), None)

    def add_node(self, title, parent):
        if parent is None:
            self.tree.create_node(title, self.node_num)
            self.node_num += 1
            return
        self.tree.create_node(title, self.node_num, parent=parent)
        self.node_num += 1

    def get_tree(self, data, results, factors, factors_values, mode, parent):
        n = len(data)
        if n < 1:
            self.add_node(mode, parent)
            return
        unique_result = True
        result = results[0]
        for i in range(1, n):
            if results[i] != result:
                unique_result = False
                break
        if unique_result:
            self.add_node(result, parent)
            return
        if len(factors) < 1:
            self.add_node(statistics.mode(results), parent)
            return
        frequency = []
        num_factors = len(factors)
        for i in range(num_factors):
            frequency.append(np.zeros((len(factors_values[i]), self.num_possible_results)))
        for i in range(len(data)):
            for j in range(len(data[i])):
                frequency[j][factors_values[j].index(data[i][j])][self.possible_results.index(results[i])] += 1
        min_entropy = 1
        argmin_entropy = 0
        for i in range(num_factors):
            entropy = 0
            for j in range(len(factors_values[i])):
                nk = np.sum(frequency[i][j])
                term = 0
                for k in range(self.num_possible_results):
                    if frequency[i][j][k] > 0:
                        term -= frequency[i][j][k] * np.log2(frequency[i][j][k] / nk) / nk
                entropy += nk * term / n
            if entropy < min_entropy:
                min_entropy = entropy
                argmin_entropy = i
        node_index = self.node_num
        self.add_node(factors[argmin_entropy], parent)
        for factors_value in factors_values[argmin_entropy]:
            new_data = []
            new_results = []
            for i in range(n):
                if data[i][argmin_entropy] == factors_value:
                    new_data.append(data[i][:])
                    new_data[-1].pop(argmin_entropy)
                    new_results.append(results[i])
            new_factors = factors[:]
            new_factors.pop(argmin_entropy)
            new_factors_values = copy.deepcopy(factors_values)
            new_factors_values.pop(argmin_entropy)
            child_index = self.node_num
            self.add_node(factors_value, node_index)
            self.get_tree(new_data, new_results, new_factors, new_factors_values, statistics.mode(results), child_index)

    def show(self):
        self.tree.show()

    def evaluate(self, data, results):
        # TODO: Receber dados e resultados, retorna um medidor de acerto
        pass
