from treelib import Tree
import copy
import numpy as np


class DecisionTree:

    def __init__(self, data, results, possible_results, factors, factors_values):
        self.node_num = 0
        self.tree = Tree()
        self.get_tree(copy.deepcopy(data), results[:], possible_results[:], factors[:], copy.deepcopy(factors_values), None)

    def add_node(self, title, parent):
        if parent is None:
            self.tree.create_node(title, self.node_num)
            self.node_num += 1
            return
        self.tree.create_node(title, self.node_num, parent=parent)
        self.node_num += 1

    def get_tree(self, data, results, possible_results, factors, factors_values, parent):
        n = len(data)
        if n < 1:
            return  # Keep no answer here?
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
            dictionary = {}
            for result in results:
                if result in dictionary.keys():
                    dictionary[result] += 1
                else:
                    dictionary[result] = 1
            self.add_node(max(dictionary, key=dictionary.get), parent)
            return
        frequency = []
        num_possible_results = len(possible_results)
        num_factors = len(factors)
        for i in range(num_factors):
            frequency.append(np.zeros((len(factors_values[i]), num_possible_results)))
        for i in range(len(data)):
            for j in range(len(data[i])):
                frequency[j][factors_values[j].index(data[i][j])][possible_results.index(results[i])] += 1
        max_entropy = 0
        argmax_entropy = 0
        for i in range(num_factors):
            entropy = 0
            for j in range(len(factors_values[i])):
                nk = np.sum(frequency[i][j])
                term = 0
                for k in range(num_possible_results):
                    term += frequency[i][j][k] / nk * np.log2(frequency[i][j][k] / nk)
                entropy += nk * term
            if entropy > max_entropy:
                max_entropy = entropy
                argmax_entropy = i
        node_index = self.node_num
        self.add_node(factors[argmax_entropy], parent)
        for factors_value in factors_values[argmax_entropy]:
            new_data = []
            new_results = []
            for i in range(n):
                if data[i][argmax_entropy] == factors_value:
                    new_data.append(data[i][:])
                    new_data[-1].pop(argmax_entropy)
                    new_results.append(results[i])
            new_factors = factors[:]
            new_factors.pop(argmax_entropy)
            new_factors_values = copy.deepcopy(factors_values)
            new_factors_values.pop(argmax_entropy)
            child_index = self.node_num
            if new_data:
                self.add_node(factors_value, node_index)
            self.get_tree(new_data, new_results, possible_results, new_factors, new_factors_values, child_index)

    def show(self):
        self.tree.show()
