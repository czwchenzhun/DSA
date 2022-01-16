# -*- coding: utf-8 -*-
# @Time    : 2022/1/16 11:12
# @Author  : Chen Zhun

import benchmarkfunctions as bf
from PSO import PSO
from DSA import DSA
from util import *


def plot(gbHistory, name):
    if name in ['Schwefel_2_26', 'ShekelFoxholes', 'Kowalik', 'SixHumpCamelBack',
                'SixHumpCamelBack', 'CrossInTray', 'DropWave', 'Eggholder']:
        plotFitness(gbHistory, name)
    else:
        plotSemilogFitness(gbHistory, name)


if __name__ == '__main__':
    bf.BenchmarkFunction.DIM = 30  # 修改任意维的测试函数的维度
    swarmNum = 50
    iterTimes = 200
    BFS = bf.BenchmarkFunction.__subclasses__()
    for BF in BFS:
        name = BF.__name__
        print("======{}======".format(name))
        func = BF()
        vran = func.xran * 0.1
        algorithm = PSO(swarmNum, func.dim, iterTimes, func, func.xran, vran)
        psoRes = algorithm.run()
        print('pso best fitness:', psoRes[0])
        algorithm = DSA(swarmNum, func.dim, iterTimes, func, func.xran)
        dsaRes = algorithm.run()
        print('dsa best fitness:', dsaRes[0])

        plot(psoRes[2], name)
        plot(dsaRes[2], name)
        plt.legend(['PSO', 'DSA'])
        plt.savefig('./figures/{}.jpg'.format(name))
        plt.show()
