import pandas as pd


def intersection_module(v1, v2):
    s = 0
    for comp in range(len(v1)):
        if v1[comp] == 1 and v2[comp] == 1:
            s += 1
    return s


def intersection_vectors(v1, v2):
    for comp in range(len(v1)):
        if v2[comp] == 1 and v2[comp] == 1:
            pass
        else:
            v1[comp] = 0
    return v1


def df_to_array(filename, header):
    with open(filename) as f:
        if header:
            head = f.readline()
        objects = list(map(lambda x: x.strip().split(','), f.readlines()))
    for i in range(len(objects)):
        for j in range(len(objects[i])):
            objects[i][j] = int(objects[i][j])

    return head, objects


filename = '/Users/anatoliy/PycharmProjects/basic_AI2/LAB_2/LAB2_AI.csv'
header = 1
ATTENTION = 0.3
B = 0.5
MAX_ITERATIONS = 0
names, input_vectors = df_to_array(filename, header)
print(names)
p = []
num_cluster = [-1] * len(input_vectors)
d = len(input_vectors[0])
p.append(input_vectors[0])
stop_iteration = 1

while stop_iteration:
    cluster_update = 0
    for i in range(len(input_vectors)):
        iteration = 0
        for j in range(len(p)):
            module_PE = intersection_module(input_vectors[i], p[j])
            iteration += 1
            if ((module_PE / (B+sum(p[j]))) > (sum(input_vectors[i]) / (B + d))) and ((module_PE / sum(input_vectors[i])) > ATTENTION):
                p[j] = intersection_vectors(p[j], input_vectors[i])
                if num_cluster[i] != j:
                    cluster_update += 1
                num_cluster[i] = j
                break
        if iteration == len(p):
            p.append(input_vectors[i])
    if cluster_update == 0:
        stop_iteration = 0
print(num_cluster)



