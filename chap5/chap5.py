#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__ = 'Kensuke Mitsuzawa'
import pandas as pd
from scipy import stats
import math


def LoadData():
    ab_test_imp = pd.read_csv('../sample_data/section5-ab_test_imp.csv', sep=",")
    ab_test_goal = pd.read_csv('../sample_data/section5-ab_test_goal.csv', sep=",")

    return ab_test_goal, ab_test_imp


def MergeData(ab_test_goal, ab_test_imp):
    ab_test_imp = pd.merge(ab_test_imp, ab_test_goal,
                           on=['transaction_id'],
                           how="left")
    print(ab_test_imp.head())

    return ab_test_imp


def MakeNewFlag(ab_test_imp):
    # python2.6以降のみで有効
    ifelse_func = lambda x: 0 if math.isnan(x) else 1
    ab_test_imp.is_copy = False
    ab_test_imp['is_goal'] = ab_test_imp['user_id_y'].apply(ifelse_func)
    print(ab_test_imp.head())

    return ab_test_imp


def Calc_func(x):
    N_click_user = len(x.is_goal[x.is_goal == 1])
    cvr = float(N_click_user) / len(x.user_id_x)
    return cvr


def SummariseClickRate(ab_test_imp):
    cvr_result = ab_test_imp.groupby('test_case_x').apply(Calc_func)
    result_map = stats.chi2_contingency(cvr_result)
    print u'-----------------------------'
    #print u'p値は{}'.format(result_map[0])


def Calc_func2(x):
    N_click_user = len(x.is_goal[x.is_goal == 1])
    cvr = float(N_click_user) / len(x.user_id_x)


def SummarisePerdate(ab_test_imp):
    # ab_test_imp_summary = ab_test_imp.groupby(
    #    ['log_date_x', 'test_case_x']).agg(
    #        {'user_id_x': {'imp': 'size'},
    #         'is_goal': {'cv': 'sum'},
    #         'aa': 'mean'})
    ab_test_imp_summary = ab_test_imp.groupby(
        ['log_date_x', 'test_case_x'])[['user_id_x', 'is_goal']].apply(Calc_func)
    ab_test_imp_summary = pd.DataFrame(ab_test_imp_summary).reset_index()
    
    ab_test_imp_sum_length_summary = ab_test_imp.groupby(
        ['log_date_x', 'test_case_x']).agg(
            {'user_id_x': {'imp': 'size'},
             'is_goal': {'cv': 'sum'}})
    ab_test_imp_sum_length_summary = pd.DataFrame(ab_test_imp_sum_length_summary).reset_index()

    ab_test_imp_sum_length_summary.columns = ['log_date_x', 'test_case_x', 'cv', 'imp']
    Calc_func2 = lambda x: float(sum(x.cv)) / sum(x.imp)
    ab_test_imp_summary = ab_test_imp_sum_length_summary.groupby('test_case_x').apply(Calc_func2)
    print ab_test_imp_summary

if __name__ == '__main__':
    ab_test_goal, ab_test_imp = LoadData()
    ab_test_imp = MergeData(ab_test_goal, ab_test_imp)
    ab_test_imp = MakeNewFlag(ab_test_imp)
    SummariseClickRate(ab_test_imp)
    SummarisePerdate(ab_test_imp)
