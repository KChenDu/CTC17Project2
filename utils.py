import csv


def get_data(file):
    num_columns = 6
    data = []
    results = []
    factors = [[], [], [], [], [], []]
    with open(file, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            if reader.line_num > 1:
                data.append([row[1] + row[2], row[3], row[5], row[6], row[7], row[8]])
                results.append(row[4])
                for i in range(num_columns):
                    if not data[-1][i] in factors[i]:
                        factors[i].append(data[-1][i])
    return data, results, factors
