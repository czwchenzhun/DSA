# -*- coding: utf-8 -*-
# @Time    : 2022/1/14 16:00
# @Author  : Chen Zhun

# Duck Swarm Algorithm (鸭群算法)
# link: https://arxiv.org/abs/2112.13508v1

import numpy as np
import random


class Duck:
    g = None
    gbest = float('+inf')

    def __init__(self, dimension, xrange, targetFunc):
        self.d = dimension
        self.xmin, self.xmax = xrange
        self.targetFunc = targetFunc
        self.x = np.random.rand(self.d) * (self.xmax - self.xmin) + self.xmin
        self.pbest = float('+inf')  # 对于这个粒子来说的最好适应度
        self.p = self.x  # 对于这个粒子来说的最好位置
        self.updateFitness()

    def clipX(self):
        self.x[self.x > self.xmax] = self.xmax
        self.x[self.x < self.xmin] = self.xmin

    def updateFitness(self):
        fitness = self.fitness()
        if fitness < self.pbest:
            self.pbest = fitness
            self.p = self.x
            if self.pbest < Duck.gbest:
                Duck.gbest = self.pbest
                Duck.g = self.p

    def fitness(self, x=None):
        if x is None:
            return self.targetFunc(self.x)
        else:
            return self.targetFunc(x)


class DSA:
    def __init__(self, particleNum, dimension, iterTimes, targetFunc, xran, P=0.5, FP=0.618):
        Duck.g = None
        Duck.gbest = float('+inf')
        self.particleNum = particleNum
        self.dimension = dimension
        self.tmax = iterTimes
        self.P = P
        self.FP = FP
        self.particles = [Duck(dimension, xran, targetFunc) for i in range(particleNum)]

    def run(self):
        self.gbestList = []
        indice = [i for i in range(len(self.particles))]
        for t in range(1, self.tmax + 1):
            K = np.sin(2 * np.random.random()) + 1
            Miu = K * (1 - t / self.tmax)
            CF1 = 1 / self.FP * np.random.random()
            CF2 = 1 / self.FP * np.random.random()
            KF1 = 1 / self.FP * np.random.random()
            KF2 = 1 / self.FP * np.random.random()
            # Exploration phase Begin
            for i, particle in enumerate(self.particles):
                if self.P > np.random.random():
                    particle.x = particle.x + Miu * particle.x * np.sign(np.random.random() - 0.5)
                else:
                    remain = list(indice)
                    remain.remove(i)
                    j = random.choice(remain)
                    particle.x = particle.x + CF1 * (Duck.g - particle.x) + CF2 * (self.particles[j].x - particle.x)
            for particle in self.particles:
                particle.clipX()
                particle.updateFitness()
            # Exploration phase End
            # ========================
            # Exploitation phase Begin
            for i, particle in enumerate(self.particles):
                x = particle.x + Miu * (Duck.g - particle.x)
                if particle.fitness(x) < particle.fitness():
                    particle.x = x
                else:
                    remain = list(indice)
                    remain.remove(i)
                    j, k = random.choices(remain, k=2)
                    particle.x = particle.x + KF1 * (Duck.g - particle.x) + KF2 * (
                            self.particles[k].x - self.particles[j].x)
            for particle in self.particles:
                particle.clipX()
                particle.updateFitness()
            self.gbestList.append(Duck.gbest)
        return Duck.gbest, Duck.g, list(self.gbestList)


if __name__ == "__main__":
    import benchmarkfunctions as bf
    from util import *

    func = bf.Ackley()
    algorithm = DSA(50, func.dim, 400, func, func.xran)
    gb, x, gbHistory = algorithm.run()
    print('best fitness:', gb)
    print('best x:', x)
    plotSemilogFitness(gbHistory, func.__class__.__name__)
    plt.show()
