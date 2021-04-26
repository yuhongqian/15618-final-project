"""
Generates Big-Bird-like sparse matrix
"""

import sys
import numpy as np
import pdb

# TODO: currently an imitation of Figure 1. Details may differ
def random_attention(seq_len, num_random):
    data = np.zeros((seq_len, seq_len))
    for i in range(seq_len):
        idxs = np.random.randint(0, seq_len, size=(1, num_random))
        data[i, idxs] = 1
    return data


def window_attention(seq_len, size):
    data = np.zeros((seq_len, seq_len))
    first_half = size // 2
    second_half = size - first_half
    for i in range(seq_len):
        start = max(0, 1 - first_half)
        end = min(len(data[i]), i + second_half)
        data[i, start:end] = 1
    return data


def global_attention(seq_len, size):
    data = np.zeros((seq_len, seq_len))
    for i in range(size):
        data[i] = np.ones_like(data[i])
    for i in range(size+1, seq_len):
        data[i, :size] = np.ones((size , 1)).reshape(-1)
    return data


def write_single_matrix(data, h, w, f):
    f.write(str(h) + " " + str(w) + "\n")
    for row in data:
        row = [str(x) for x in row]
        f.write(' '.join(row) + "\n")


def write_matrix(seq_len, batch_size=32, r_size=2, w_size=3, g_size=2):
    output_file = "data/seqlen{}_bs{}_r{}_w{}_g{}".format(seq_len, batch_size, r_size, w_size, g_size)
    attention = np.logical_or(np.logical_or(random_attention(seq_len, r_size), window_attention(seq_len, w_size)),
                              global_attention(seq_len, g_size))
    # pdb.set_trace()
    data = np.random.randn(seq_len, seq_len)
    data = data * attention
    fp = open(output_file, "w")
    write_single_matrix(data, seq_len, seq_len, fp)
    data = np.random.randn(seq_len, batch_size)
    write_single_matrix(data, seq_len, batch_size, fp)
    fp.close()

if __name__ == '__main__':
    # r = int(sys.argv[1])
    write_matrix(seq_len=4, batch_size=2, r_size=2, w_size=2, g_size=2)






