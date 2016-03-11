import numpy as np

vals = [0.2,0.01,0.5,0.1,0.1]

def get_random(row):
    rand_prob = np.random.uniform(0.0,1.0)
    print rand_prob
    cumulative_prob = 0
    for idx in range(len(row)):
        cumulative_prob += row[idx]
        if rand_prob <= cumulative_prob:
            return idx
    return row.index(max(row))
print get_random(vals)