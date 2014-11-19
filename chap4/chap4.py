#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__ = 'Kensuke Mitsuzawa'
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime as dt

def LoadCSV():
    dau = pd.read_csv('../sample_data/section4-dau.csv', sep=",")    
    user_info = pd.read_csv('../sample_data/section4-user_info.csv', sep=",")
    return dau, user_info


def MergeData(dau, user_info):
    dau_user_info = pd.merge(dau,
             user_info,
             on=['user_id', 'app_name'],
             how='inner')
    return dau_user_info


def SegmentSummarise(dau_user_info):
    dau_user_info.is_copy = False
    substr_func = lambda x: x[0:8]
    dau_user_info['log_month'] = dau_user_info['log_date'].apply(substr_func) 
    print(pd.crosstab(dau_user_info['log_month'], dau_user_info['gender']))

    print(pd.crosstab(dau_user_info['log_month'], dau_user_info['generation']))

    pivot_table = dau_user_info.pivot_table(values='user_id',
                                          index='log_month',
                                          columns=['gender', 'generation'],
                                          fill_value=0)
    print(pivot_table)

    # デバイスごと
    print(pd.crosstab(dau_user_info['log_month'], dau_user_info['device_type']))


def PlotResult(dau_user_info):
    """
    結果を時系列データでプロットする
    """
    dau_user_info_device_summary = dau_user_info.groupby(
        ['log_date', 'device_type']).apply(lambda x: len(x.user_id)).reset_index()
    # pandasを用いたグラフでグループ化する場合は、列ごとにグループ化する必要がある
    graph_input = pd.pivot_table(dau_user_info_device_summary,
                                 index='log_date',
                                 columns='device_type', values=0).reset_index()
    graph_input['log_date'] = graph_input['log_date'].apply(lambda x: dt.strptime(x, '%Y-%m-%d'))

    plt.figure()
    graph_input.plot()
    plt.show()
    plt.savefig('./dau_user_summary.png')

if __name__ == '__main__':
    dau, user_info = LoadCSV()
    dau_user_info = MergeData(dau, user_info)
    SegmentSummarise(dau_user_info)
