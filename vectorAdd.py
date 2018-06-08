import numpy as np
from timeit import default_timer as time
from numba import vectorize

@vectorize(["float32(float32, float32)", target='gpu'])
def vectorAdd(x, y):
	return x + y
def main():
        N = 50000000
        a1 = np.ones(N, dtype=np.float32)
        a2 = np.ones(N, dtype=np.float32)
        a3 = np.zeros(N, dtype=np.float32)
        start = time()

        z = vectorAdd(a1, a2)

        finalTime = time() - start

        print("Add took, %f", finalTime)

if __name__ == '__main__':
        main()