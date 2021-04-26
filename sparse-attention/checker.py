import sys
import numpy as np


def main(input_file):
    with open(input_file, "r") as f:
        m, k = f.readline().split()
        m, k = int(m), int(k)
        A = []
        for i in range(m):
            row = [float(x) for x in f.readline().split()]
            A.append(row)
        _, _ = f.readline().split()
        B = []
        for i in range(k):
            row = [float(x) for x in f.readline().split()]
            B.append(row)
    print(np.dot(np.array(A), np.array(B)))


if __name__ == "__main__":
    main(sys.argv[1])

