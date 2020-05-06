from multiprocessing import Pool
import time
import matplotlib.pyplot as plt


def sum_square(number):
    a = 0
    for i in range(number):
        a += i**2
    return a

def multi(numbers):
 
    start = time.time()
    p = Pool()
    result = p.map(sum_square, numbers)
    
    p.close()
    p.join()
    total_time = time.time() - start

    return total_time

def single(numbers):
    start = time.time()
    result = []
    for i in numbers:
        result.append(sum_square(i))
    total_time = time.time() - start

    return total_time

MP = []
N = []
if __name__ == "__main__":
    for i in range(20000):
        A = range(i)
        print(i)
        MP.append(multi(A))
        N.append(single(A))
    plt.plot(MP)
    plt.plot(N)
    plt.legend(["Multiprocess", "Normal"])
    plt.savefig("Test")
