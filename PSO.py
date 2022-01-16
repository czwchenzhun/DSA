# -*- coding: utf-8 -*-
# @Time    : 2022/1/15 9:37
# @Author  : Chen Zhun
# 惯性权重粒子群算法
import matplotlib.pyplot as plt
import numpy as np


class Particle:
    g = None
    gbest = float('+inf')

    def __init__(self, dimension, xrange, vrange, targetFunc):
        self.d = dimension
        self.xmin, self.xmax = xrange
        self.vmin, self.vmax = vrange
        self.targetFunc = targetFunc
        self.x = np.random.rand(self.d) * (self.xmax - self.xmin) + self.xmin
        self.v = np.random.rand(self.d) * (self.vmax - self.vmin) + self.vmin
        self.pbest = float('+inf')  # 对于这个粒子来说的最好适应度
        self.p = self.x  # 对于这个粒子来说的最好位置

    def updateX(self):
        self.x = self.x + self.v
        self.x[self.x > self.xmax] = self.xmax
        self.x[self.x < self.xmin] = self.xmin

    def updateFitness(self):
        fitness = self.fitness()
        if fitness < self.pbest:
            self.pbest = fitness
            self.p = self.x
            if self.pbest < Particle.gbest:
                Particle.gbest = self.pbest
                Particle.g = self.p

    def updateV(self, w, c1, c2):
        self.v = w * self.v + c1 * np.random.rand(self.d) * (self.p - self.x) + c2 * np.random.rand(self.d) * (
                    Particle.g - self.x)
        self.v[self.v > self.vmax] = self.vmax
        self.v[self.v < self.vmin] = self.vmin

    def fitness(self):
        return self.targetFunc(self.x)

    def run(self, w, c1, c2):
        self.updateFitness()
        self.updateX()
        self.updateV(w, c1, c2)


class PSO:
    def __init__(self, particleNum, dimension, iterTimes, targetFunc, xran, vran, w=0.8, c1=1.5, c2=1.5):
        Particle.g = None
        Particle.gbest = float('+inf')
        self.particleNum = particleNum
        self.dimension = dimension
        self.iterTimes = iterTimes
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.particles = [Particle(dimension, xran, vran, targetFunc) for i in range(particleNum)]

    def run(self):
        self.gbestList = []
        for i in range(self.iterTimes):
            w = self.w - i * (0.5 / self.iterTimes)
            for particle in self.particles:
                particle.run(w, self.c1, self.c2)
            self.gbestList.append(Particle.gbest)
        return Particle.gbest, Particle.g, list(self.gbestList)


if __name__ == "__main__":
    import benchmarkfunctions as bf
    from util import *

    func = bf.Sphere()
    vran = func.xran * 0.1
    algorithm = PSO(50, func.dim, 400, func, func.xran, vran)
    gb, x, gbHistory = algorithm.run()
    print('best fitness:', gb)
    print('best x:', x)
    plotSemilogFitness(gbHistory, func.__class__.__name__)
    plt.show()
