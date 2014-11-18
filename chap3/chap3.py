#! /usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def Page53():
    dau = pd.read_csv('../sample_data/section3-dau.csv', sep=',')
    dpu = pd.read_csv('../sample_data/section3-dpu.csv', sep=',')
    install = pd.read_csv('../sample_data/section3-install.csv', sep=',')    

    return dau, install, dpu


def Page55_56(dau, install, dpu):
    dau_install = pd.merge(dau,
                           install,
                           on=['user_id', 'app_name'])
    print (dau_install.head())

    dau_install_payment = pd.merge(dau_install,
                                   dpu,
                                   on=['log_date', 'app_name', 'user_id'],
                                   how='left')
    dau_install_payment.head()
    print(dau_install_payment.dropna().head())

    dau_install_payment = dau_install_payment.fillna(0)
    print(dau_install_payment.head())
    return dau_install_payment


def Page58(dau_install_payment):
    substr_func = lambda x: x[0:7]
    dau_install_payment.is_copy = False
    dau_install_payment['log_month'] = dau_install_payment['log_date'].apply(substr_func)
    # print(dau_install_payment.head())

    dau_install_payment['install_month'] = dau_install_payment['install_date'].apply(substr_func)
    mau_payment = dau_install_payment.groupby(['log_month', 'user_id', 'install_month']).sum()
    # groupbyでグループ化しても、インデックスになってるいるだけなので、下の処理が必要
    mau_payment = pd.DataFrame(mau_payment).reset_index()
    mau_payment.to_csv("tmp.csv")

    return mau_payment


def Ifelse_func_install_log(x):
    if x.install_month == x.log_month:
        return 'install'
    else:
        return 'existing'

def Page60_61(mau_payment):
    # axis=1で行単位の処理させる
    mau_payment['user_type'] = mau_payment.apply(Ifelse_func_install_log, axis=1)
    mau_payment_summary = mau_payment.groupby(['log_month', 'user_type']).sum()
    mau_payment_summary = pd.DataFrame(mau_payment_summary).reset_index()

    return mau_payment_summary


def PlotGraph(mau_payment_summary, mau_payment):
    #df_bar = mau_payment_summary.ix[:, ['log_month', 'payment', 'user_type']]
    #payment_bar = df_bar.plot(kind='bar', stacked=True)
    #bar_fig = payment_bar.get_figure()
    #bar_fig.savefig('./payment_bar.png')

    # matplotlibだと、次の様に書く
    w = 0.4
    # existingとinstallを別のグループにして棒グループを描く
    Y1 = mau_payment_summary.ix[mau_payment_summary.user_type == 'existing', 'payment']
    Y2 = mau_payment_summary.ix[mau_payment_summary.user_type == 'install', 'payment']
    X = np.arange(len(Y1))

    fig = plt.figure()
    fig_bar = fig.add_subplot(111)
    plt.xticks(X, ['2013-06', '2013-07'])
    fig_bar.bar(X, Y1, color='b', width=w, label='existing', align='center')
    fig_bar.bar(X, Y2, color='g', width=w, bottom=Y1, label='existing', align='center')
    fig_bar.grid(True)
    plt.show()
    plt.savefig('./bar_sum.png')

    # グループ化して、棒グラフをプロットする
    df_install_above = mau_payment[(mau_payment.payment > 0) & (mau_payment.user_type == 'install')]
    df_06 = df_install_above.ix[df_install_above.log_month == '2013-06', 'payment']
    df_07 = df_install_above.ix[df_install_above.log_month == '2013-07', 'payment']
    df_06.hist(label='2014-06')
    df_07.hist(label='2014-07')
    plt.xlabel('payment')
    plt.ylabel('count')
    plt.savefig('./hist_payment.png')
    plt.show()



def Main():
    dau, install, dpu = Page53()
    dau_install_payment = Page55_56(dau, install, dpu)
    mau_payment = Page58(dau_install_payment)
    mau_payment_summary = Page60_61(mau_payment)
    print mau_payment_summary
    PlotGraph(mau_payment_summary, mau_payment)

if __name__ == '__main__':
    Main()
