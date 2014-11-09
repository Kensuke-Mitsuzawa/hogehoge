#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__ = 'kensuke-mi'

import numpy as np
from scipy import linalg as LA
import pandas
import matplotlib.pyplot as plt
import statsmodels.api as sm


def MultiRegression(ad_data):
    """
    説明変数をtvcmとmagazine
    目的変数をinstall
    にして重回帰を実行する
    """
    print ad_data
    X = ad_data.ix[:, ['tvcm', 'magazine']]
    X = sm.add_constant(X, prepend=False)

    Y = ad_data['install']
    model = sm.OLS(Y, X)
    results = model.fit()
    print results.summary()


def LingAlgTest():
    """
    LA.solveには以下の形式のnumpy.arrayを入力する
    training data
    array([[x_a1_1, x_a2_1, x_a3_1],
    [x_a1_2, x_a2_2, x_a3_3]])

    answer
    array([y_1 y_2])
    """
    x = [9.83, -9.97, -3.91, -3.94, -13.67, -14.04, 4.81, 7.65, 5.50, -3.34]
    y = [-5.50, -13.53, -1.23, 6.07, 1.94, 2.79, -5.43, 15.57, 7.26, 1.34]
    z = [635.99, 163.78, 86.94, 245.35, 1132.88,
         1239.55, 214.01, 67.94, -1.48, 104.18]

    N = len(x)
    G = np.array([x, y, np.ones(N)]).T
    result = LA.solve(G.T.dot(G), G.T.dot(z))

    return result


def LoadData():
    DATA_PATH = "../sample_data/ad_result.csv"
    ad_data = pandas.io.parsers.read_csv(
        filepath_or_buffer=DATA_PATH, sep=",")

    return ad_data


def PlotTvcmInstall(ad_data):
    """
    広告費と新規ユーザー数の散布図を作る
    """
    # ipythonで対話的に実行するときは以下のコード
    # plt.scatter(ad_data['tvcm'], ad_data['install'])
    # plt.savefig('./tvcm_install.png')
    fig = plt.figure()
    ax = fig.add_subplot(111)
    sc = ax.scatter(ad_data['tvcm'], ad_data['install'])
    ax.set_title(u'')
    ax.set_xlabel(u'TVの広告費')
    ax.set_ylabel(u'新規インストール')
    ax.grid(True)
    # plt.show()
    plt.savefig('./tvcm_install.png')

def PlotMagazineInstall(ad_data):
    """
    雑誌の広告費と新規インストール数の散布図を作成する
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    scatter = ax.scatter(ad_data['magazine'], ad_data['install']) 
    ax.set_title('')
    ax.set_xlabel(u'雑誌の広告費')
    ax.set_ylabel(u'インストール数')
    ax.grid(True)
    # plt.show()
    plt.savefig('./magazine_install.png')


def Main():
    ad_data = LoadData()
    PlotTvcmInstall(ad_data)
    PlotMagazineInstall(ad_data)
    MultiRegression(ad_data)

if __name__ == '__main__':
    Main()
