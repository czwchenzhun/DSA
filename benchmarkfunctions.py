# -*- coding: utf-8 -*-
# @Time    : 2022/1/14 15:11
# @Author  : Chen Zhun

import numpy as np


class BenchmarkFunction:
    DIM = 10

    def __init__(self):
        pass

# 任意维单峰测试函数
class Sphere(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-100, 100])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        return np.sum(x ** 2)

    def __str__(self):
        desc = "Sphere Function : min(F(x)) = F(0,...,0) = 0"
        return desc


class Schwefel_2_22(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-10, 10])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        total = np.sum(np.abs(x)) + np.prod(np.abs(x))
        return total

    def __str__(self):
        desc = "Schwefel's Problem 2.22 : min(F(x)) = F(0,...,0) = 0"
        return desc


class Schwefel_1_2(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-100, 100])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        D = len(x)
        total = 0
        for i in range(D):
            v = np.sum(x[:i])
            total += v ** 2
        return total

    def __str__(self):
        desc = "Schwefel's Problem 1.2 : min(F(x)) = F(0,...,0) = 0"
        return desc


class Schwefel_2_21(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-100, 100])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        return np.max(np.abs(x))

    def __str__(self):
        desc = "Schwefel's Problem 2.21 : min(F(x)) = F(0,...,0) = 0"
        return desc


class Rosenbrock(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-30, 30])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        D = len(x)
        total = 0
        for i in range(D - 1):
            v = 100 * ((x[i] ** 2 - x[i + 1]) ** 2) + (x[i] - 1) ** 2
            total += v
        return total

    def __str__(self):
        desc = "Generalized Rosenbrock's Function : min(F(x)) = F(1,...,1) = 0"
        return desc


class Step(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-100, 100])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        return np.sum(np.floor(x + 0.5) ** 2)

    def __str__(self):
        desc = "Step Function : min(F(x)) = F(0,...,0) = 0"
        return desc


class Quartic(BenchmarkFunction):
    '''
    Quartic Function i.e. Noise
    '''

    def __init__(self):
        self.xran = np.array([-1.28, 1.28])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        D = len(x)
        total = 0
        for i in range(D):
            total += (i + 1) * x[i] ** 4
        return total + np.random.random()

    def __str__(self):
        desc = "Quartic/Noise Function : min(F(x)) = F(0,...,0) 趋近于rand[0,1)"
        return desc


# 任意维多峰峰测试函数
class Schwefel_2_26(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-500, 500])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        return -np.sum(x * np.sin(np.sqrt(np.abs(x))))

    def __str__(self):
        desc = "Generalized Schwefel's Problem 2.26 : min(F(x)) = F(420.9687,...,420.9687) = -(dim*418.9828872721625)"
        return desc


class Rastrigin(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-5.12, 5.12])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        v = x ** 2 - 10 * np.cos(2 * np.pi * x)
        return 10 * len(x) + np.sum(v)

    def __str__(self):
        desc = "Generalized Rastrigin's Function : min(F(x)) = F(0,...,0) = 0"
        return desc


class Ackley(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-32, 32])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        a = 20
        b = 0.2
        c = 2 * np.pi
        d = a + np.exp(1)
        v = np.mean(np.cos(c * x))
        res = -a * np.exp(-b * np.linalg.norm(x)) - np.exp(v) + d
        return res

    def __str__(self):
        desc = "Ackley's Function : min(F(x)) = F(0,...,0) = 0"
        return desc


class Griewank(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-600, 600])
        self.dim = BenchmarkFunction.DIM

    def __call__(self, x):
        a = np.sum(x ** 2) / 4000
        b = 1
        for i in range(len(x)):
            b *= np.cos(x[i] / np.sqrt(i + 1))
        return a - b + 1

    def __str__(self):
        desc = "Generalized Griewank's Function : min(F(x)) = F(0,...,0) = 0"
        return desc

# 固定维多峰测试函数
class ShekelFoxholes(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-65.536, 65.536])
        self.dim = 2
        self.aS = [
            [-32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32, -32, -16, 0, 16, 32],
            [-32, -32, -32, -32, -32, -16, -16, -16, -16, -16, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32]]
        self.aS = np.array(self.aS)

    def __call__(self, x):
        bS = np.zeros((25))
        for j in range(25):
            bS[j] = np.sum((x - self.aS[:, j]) ** 6) + j + 1
        return (0.002 + np.sum(1 / bS)) ** -1

    def __str__(self):
        desc = "Shekel's Foxholes Function : min(F(x)) = F(-32,-32) = 0.7754 ≈ 1"
        return desc


class Kowalik(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-5, 5])
        self.dim = 4
        self.a = np.array([.1957, .1947, .1735, .16, .0844, .0627, .0456, .0342, .0323, .0235, .0246])
        self.b = np.array([.25, .5, 1, 2, 4, 6, 8, 10, 12, 14, 16])
        self.b = 1 / self.b
        self.b_2 = self.b ** 2

    def __call__(self, x):
        p1 = x[0] * (self.b_2 + self.b * x[1])
        p2 = self.b_2 + self.b * x[2] + x[3]
        return np.sum((self.a - p1 / p2) ** 2)

    def __str__(self):
        desc = "Kowalik's Function : min(F(x)) ≈ F(0.1928,0.1928,0.1231,0.1358) ≈ 0.0003075"
        return desc


class SixHumpCamelBack(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-5, 5])
        self.dim = 2

    def __call__(self, x):
        return 4 * x[0] ** 2 - 2.1 * x[0] ** 4 + 1 / 3 * x[0] ** 6 + x[0] * x[1] - 4 * x[1] ** 2 + 4 * x[1] ** 4

    def __str__(self):
        desc = "Six-Hump Camel-Back Function : min(F(x)) = F(0.08983,-0.7126) = F(-0.08983,0.7126) = -1.0316285"
        return desc


class CrossInTray(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-10, 10])
        self.dim = 2

    def __call__(self, x):
        p = np.abs(100 - np.sqrt(x[0] ** 2 + x[1] ** 2) / np.pi)
        p = np.sin(x[0]) * np.sin(x[1]) * np.exp(p)
        p = np.abs(p) + 1
        return -0.0001 * p ** 0.1

    def __str__(self):
        desc = "Cross-in-Tray Function : min(F(x)) = F(1.3491,-1.3491) = F(1.3491,1.3491) = F(-1.3491,1.3491) = F(-1.3491,-1.3491) = -2.06261"
        return desc


class DropWave(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-5.12, 5.12])
        self.dim = 2

    def __call__(self, x):
        a = x[0] ** 2 + x[1] ** 2
        return -(1 + np.cos(12 * np.sqrt(a))) / (0.5 * a + 2)

    def __str__(self):
        desc = "Drop-Wave Function : min(F(x)) = F(0,0) = -1"
        return desc


class Eggholder(BenchmarkFunction):
    def __init__(self):
        self.xran = np.array([-512, 512])
        self.dim = 2

    def __call__(self, x):
        p1 = -(x[1] + 47) * np.sin(np.sqrt(np.abs(x[1] + x[0] / 2 + 47)))
        p2 = -x[0] * np.sin(np.sqrt(np.abs(x[0] - (x[1] + 47))))
        return p1 + p2

    def __str__(self):
        desc = "Eggholder Function : min(F(x)) = F(512,404.2319) = -959.6407"
        return desc
