# -*- coding: utf-8 -*-
# @Time    : 2022/1/16 10:11
# @Author  : Chen Zhun

import matplotlib.pyplot as plt


def plotFitness(ls, title=""):
    plt.title("Fitness Evolution Curve: " + title)
    plt.xlabel("iterations")
    plt.ylabel("fitness")
    x = [i for i in range(len(ls))]
    plt.plot(x, ls)


def plotSemilogFitness(ls, title=""):
    plt.title("Semilog Fitness Evolution Curve: " + title)
    plt.xlabel("iterations")
    plt.ylabel("semilog fitness")
    plt.semilogy(ls)
