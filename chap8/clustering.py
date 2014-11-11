#! /usr/bin/python
# -*- coding:utf-8 -*-
import datetime
import pandas as pd
from sklearn.cluster import KMeans


def ReadTsvDates(base_dir, app_name, date_from, date_to):
    date_from = datetime.datetime.strptime(date_from, '%Y-%m-%d')
    date_to = datetime.datetime.strptime(date_to, '%Y-%m-%d')
    # 先に日付の系列をつくってしまう
    date_seq = []
    day_tmp = date_from
    if date_from == date_to:
        date_seq.append(date_from.strftime('%Y-%m-%d'))
    else:
        while True:
            date_seq.append(day_tmp.strftime('%Y-%m-%d'))
            day_tmp = day_tmp + datetime.timedelta(1)
            if day_tmp == date_to:
                date_seq.append(day_tmp.strftime('%Y-%m-%d'))
                break

    # xは先に宣言しておく
    x = pd.DataFrame()
    for date in date_seq:
        data_path = '{}/{}/{}/data.tsv'.format(base_dir, app_name, date)
        loaded_data = pd.io.parsers.read_csv(
            filepath_or_buffer=data_path, sep="\t")
        x = x.append(loaded_data)
    return x


def ReadDau(app_name, date_from, date_to):
    DATA_PATH = '../sample_data/sample-data/section8/daily/dau'
    data = ReadTsvDates(DATA_PATH, app_name, date_from, date_to)

    return data


def ReadDpu(app_name, date_from, date_to):
    DATA_PATH = '../sample_data/sample-data/section8/daily/dpu'
    data = ReadTsvDates(DATA_PATH, app_name, date_from, date_to)

    return data


def ReadActionDaily(app_name, date_from, date_to):
    DATA_PATH = '../sample_data/sample-data/section8/daily/action/'
    data = ReadTsvDates(DATA_PATH, app_name, date_from, date_to)

    return data


def PrepareNewDPU(dau, dpu):
    dau2 = pd.merge(dau,
                    dpu.ix[:, ['log_date', 'user_id', 'payment']],
                    on=['log_date', 'user_id'], how='left')
    dau2 = dau2.fillna(0)

    dau2.is_copy = False
    # 最新バージョンを除いて、ifelseに相当するものがないので、匿名関数で定義する
    iflese_func = lambda x: 1 if(x['payment' != 0]) else 0
    dau2['is_payment'] = dau2['payment'].apply(iflese_func)

    return dau2


def PrepareMauDate(dau2):
    dau2.is_copy = False

    substr_func = lambda x: x[0:7]
    dau2['log_month'] = dau2['log_date'].apply(substr_func)

    """
    複数の列名を新しく作成する場合は以下の表記
    グループ化したオブジェクト.agg({'既存の列名': {'新しく名づけたい列名': '関数名(メソッド名)'},
    '既存の列名２': {'新しい列名２': '関数（メソッド名）'}})
    """
    mau = dau2.groupby(['log_month', 'user_id']).agg(
        {'payment': {'payment': 'sum'}, 'log_date': {'access_days': 'size'}})

    return mau


def CallKmeans(input_array, NUM_CLUSTERS=10):
    km = KMeans(n_clusters=NUM_CLUSTERS,
                init='k-means++', n_init=1, verbose=True)  # Kmeansインスタンスを作成
    km.fit(input_array)  # 実データにfitting
    labels = km.labels_  # 各要素にクラス番号をふる  戻り値はnumpy.ndarray

    return labels


def DokMenasClusteringFrist(user_action):
    # dfからarrayに変換する
    df_clustering_input = user_action.ix[:, ['A47']]
    user_action_array = df_clustering_input.as_matrix()
    # ここにクラスタリングを実行するコードを記述
    array_cluster_label = CallKmeans(user_action_array, 3)
    series_cluster_label = pd.Series(array_cluster_label)
    # padnasではseriesに対して要素カウントするときは、series.value_counts()を使う
    print (series_cluster_label.value_counts())
    df_cluster_label = pd.DataFrame(array_cluster_label)
    df_cluster_label.columns = ['cluster_label']
    # cbindの操作
    user_action = pd.concat([user_action, df_cluster_label], axis=1)
    return user_action


def CallPCA():
    """
    statsmodelからPCAを呼ぶ
    """


def DoPCA():
    """
    PCAを実行する
    """
    # 情報量が０に近い変数の削除

    # 変数間の相関が高い変数の削除 


def Main():
    """
    dau = ReadDau('game-01', '2013-10-01', '2013-10-31')
    dpu = ReadDpu('game-01', '2013-10-01', '2013-10-31')
    dau2 = PrepareNewDPU(dau, dpu)
    mau = PrepareMauDate(dau2)
    """
    user_action = ReadActionDaily('game-01', '2013-10-31', '2013-10-31')
    user_action = DokMenasClusteringFrist(user_action)
    user_action_h = user_action[user_action.cluster_label >= 1]
    # ランキングポイント分布を描画すること
    # 後処理コードを記述
    # PCAするコードを記述


if __name__ == '__main__':
    # base_dir = '../sample_data/sample-data/section8/daily/dau/'
    # print ReadTsvDates(base_dir, 'game-01', '2013-05-01', '2013-10-31')
    Main()
