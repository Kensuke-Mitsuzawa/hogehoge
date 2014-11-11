#! /usr/bin/python
# -*- coding:utf-8 -*-
import datetime
import pandas as pd
import pyper as pr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


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
                init='k-means++', n_init=10,
                verbose=True, max_iter=100)  # Kmeansインスタンスを作成
    km.fit(input_array)  # 実データにfitting
    labels = km.labels_  # 各要素にクラス番号をふる  戻り値はnumpy.ndarray

    return labels


def Callykmeans(input_df):

    r = pr.R(use_pandas="True")  # Rのインスタンスをつくる
    r.assign("user.action", input_df)  # Rにdfを渡す

    r('if ("Unnamed: 0" %in% names(user.action)){user.action$"Unnamed: 0" <- NA}')
    r('library(ykmeans)')
    r('user.action2 <- ykmeans(user.action, "A47", "A47", 3)')
    # print(r.get('table(user.action2$cluster)'))
    r('user.action.h <- user.action2[user.action2$cluster >= 2,\
      names(user.action)]')
    user_action_h = r.get('user.action.h')
    # Rの出力だと、列名にスペースが入る可能性があるので、削除
    user_action_h.columns = [title_name.strip() for title_name in user_action_h.columns]

    return user_action_h


def DokMenasClusteringFrist(user_action):
    FLAG = 'ykmeans'
    # dfからarrayに変換する
    if FLAG == 'sklearn':
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
        user_action2 = pd.concat([user_action, df_cluster_label], axis=1)
        user_action_h = user_action[user_action2.cluster_label >= 1]

    elif FLAG == 'ykmeans':
        user_action_h = Callykmeans(user_action)

    return user_action_h


def CallPCA():
    """
    statsmodelからPCAを呼ぶ
    """


def DoPreprocess(user_action_h):
    """
    PCAを実行する
    前処理部分は該当する関数がpythonにはないので、簡便のため、Rを呼び出す
    """
    # ここにあとちょっとだけ前処理
    # print (user_action_h.columns)
    user_action_f = user_action_h.drop(
        ['log_date', 'app_name', 'user_id', 'A1'], axis=1)
    # user_action_f.index.name = user_action_h['user_id']
    print (user_action_f.shape)

    r = pr.R(use_pandas="True")  # Rのインスタンスをつくる
    r.assign("user.action.f", user_action_f)  # Rにdfを渡す
    r('if ("Unnamed: 0" %in% names(user.action.f)){user.action.f$"Unnamed: 0" <- NA}')
    # r('user.action.f = user.action.f[, -1]')  # Rに渡された状態では、インデックスの１列が邪魔なので、消す
    r("library(caret)")
    r("nzv = caret::nearZeroVar(user.action.f)")  # 情報量が０に近い変数の削除
    r("user.action.f.filterd = user.action.f[,-nzv]")
    # print r.get("ncol(user.action.f.filterd)")

    # 変数間の相関が高い変数の削除
    r("user.action.cor = cor(user.action.f.filterd)")
    r("highly.cor.f = caret::findCorrelation(user.action.cor, cutoff=.7)")
    # print r.get("highly.cor.f")
    r("user.action.f.filterd = user.action.f.filterd[,-highly.cor.f]")
    # print r.get("ncol(user.action.f.filterd)")

    user_action_f_filtered = r.get("user.action.f.filterd")
    user_action_f_filtered.columns = [
        title_name.strip() for title_name in user_action_f_filtered.columns]

    user_action_f_filtered.to_csv('./user_action_f_filtered.csv')
    return user_action_f_filtered


def DoPCA(user_action_f_filtered):
    array_user_action_f_filtered = user_action_f_filtered.values
    print array_user_action_f_filtered.shape
    pca = PCA(n_components=20)
    pca.fit(array_user_action_f_filtered)
    X_pca = pca.transform(array_user_action_f_filtered)

    df_pca = pd.DataFrame(X_pca)

    return df_pca


def DoClusteringSecond(df_pca):

    keys = ['X{}'.format(col_name) for col_name in df_pca.columns]
    r = pr.R(use_pandas="True")  # Rのインスタンスをつくる
    r.assign("user.action.pca", df_pca)  # Rにdfを渡す
    r.assign('keys', keys)
    r('library(ykmeans)')
    r('if ("Unnamed: 0" %in% names(user.action.pca)){user.action.pca$"Unnamed: 0" <- NA}')
    r('user.action.km = ykmeans(user.action.pca, keys, "X1", 3:6)')
    user_action_pca_km = r.get('user.action.km')
    user_action_pca_km.columns = [
        title_name.strip() for title_name in user_action_pca_km.columns]
    print (user_action_pca_km['cluster'].value_counts())

    return user_action_pca_km


def CalcAveragePerCluster(user_action_f_filtered, user_action_km):
    """
    クラスタごとに平均値を算出する p193
    """
    user_action_f_filtered['cluster'] = user_action_km['cluster']
    # クラスタごとにすべての軸で平均値を求める
    # 横軸はクラスタ番号で、縦軸が軸番号
    cluster_list = list(set(user_action_f_filtered['cluster']))
    df_stack = pd.DataFrame()
    for cluster_id in cluster_list:
        rows_in_cluster = user_action_f_filtered[
            user_action_f_filtered.cluster == cluster_id]
        rows_in_cluster = rows_in_cluster.drop('cluster', axis=1)
        df_mean = rows_in_cluster.mean()
        df_stack = df_stack.append(df_mean.T, ignore_index=True)

    return df_stack


def CalcKPIperCluster(user_action_f_filtered, user_action_km):
    """
    クラスタごとにKPIを求める
    ここで、useridが必要になる
    """
    user_action_f_filtered['cluster'] = user_action_km['cluster']

    pass

def Main():
    """
    dau = ReadDau('game-01', '2013-10-01', '2013-10-31')
    dpu = ReadDpu('game-01', '2013-10-01', '2013-10-31')
    dau2 = PrepareNewDPU(dau, dpu)
    mau = PrepareMauDate(dau2)
    """
    # user_action = ReadActionDaily('game-01', '2013-10-31', '2013-10-31')
    # user_action_h = DokMenasClusteringFrist(user_action)
    # user_action_h.to_csv('./user_action_h.csv')
    # user_action_h = pd.read_csv('./user_action_h.csv')
    # ランキングポイント分布を描画すること
    # user_action_f_filtered = DoPreprocess(user_action_h)
    # 実はなぜか軸が１つ多い・・・
    user_action_f_filtered = pd.read_csv('./user_action_f_filtered.csv')
    #df_pca = DoPCA(user_action_f_filtered)
    #user_action_km = DoClusteringSecond(df_pca)
    #user_action_km.to_csv('./user_action_km.csv')
    user_action_km = pd.read_csv('./user_action_km.csv')
    df_mean_per_cluster = CalcAveragePerCluster(user_action_f_filtered, user_action_km)
    CalcKPIperCluster(user_action_f_filtered, user_action_km)
    print df_mean_per_cluster


if __name__ == '__main__':
    # base_dir = '../sample_data/sample-data/section8/daily/dau/'
    # print ReadTsvDates(base_dir, 'game-01', '2013-05-01', '2013-10-31')
    Main()
