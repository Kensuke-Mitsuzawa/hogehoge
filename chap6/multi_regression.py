#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__ = 'kensuke-mi'

import numpy as np
from scipy import linalg as LA
import pandas

"""
manual of scipy.linalg.solve is below link
http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.linalg.solve.html
"""
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
    z = [635.99, 163.78, 86.94, 245.35, 1132.88, 1239.55, 214.01, 67.94, -1.48, 104.18]

    N = len(x)
    G_temp = np.array([x, y])
    G = np.array([x, y, np.ones(N)]).T
    result = LA.solve(G.T.dot(G), G.T.dot(z))

    return True


def LoadData():
    DATA_PATH = "../sample_data/ad_result.csv"
    ad_data = pandas.io.parsers.read_csv(filepath_or_buffer=DATA_PATH,
                               sep=",",
                               header=True)
    print ad_data

def Main():
    LoadData()

if __name__=='__main__':
    Main()
