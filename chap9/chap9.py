#! /usr/bin/python
# -*- coding:utf-8 -*-

import os
import pandas as pd
import datetime
import numpy as np
from sklearn import cluster
from sklearn.decomposition import PCA

from sklearn import tree



def ReadInstall(app_name, target_day):

    base_dir = '../sample_data/sample-data/section9/snapshot/install'
    # f = '{}/{}/{}/install.csv'.format(base_dir, app_name, target_day)
    f = os.path.join(base_dir, app_name, target_day, 'install.csv')
    df_Loaded = pd.read_csv(f, sep=",")

    return df_Loaded


def ReadSeqDate(app_name, date_from, date_to, f_type, base_dir, action_name=''):
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

    if f_type == 'action':
        data_path_base = '{}/{}/{}/'.format(base_dir, app_name,
                                            action_name)
        f_name = '{}.csv'.format(action_name)

    elif f_type == 'dau':
        data_path_base = '{}/{}/'.format(base_dir, app_name)
        f_name = 'dau.csv'

    x = pd.DataFrame()
    for date in date_seq:
        # data_path = '{}/{}/{}'.format(data_path_base, date, f_name)
        data_path = os.path.join(data_path_base, date, f_name)
        loaded_data = pd.io.parsers.read_csv(
            filepath_or_buffer=data_path, sep=",")
        x = x.append(loaded_data)

    return x


def __check(x):
    print type(x)
    print x


def GetDaysDiff(input_row):

    date_conv_func = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')

    if type(input_row['log_date']) == str:
        log_date = date_conv_func(input_row['log_date'])
    else:
        log_date = input_row['log_date']

    if type(input_row['log_date_inst']) == str:
        log_date_inst = date_conv_func(input_row['log_date_inst'])
    else:
        log_date_inst = input_row['log_date_inst']

    input_row.is_copy = False
    elapsed_days = log_date - log_date_inst
    Series_days_diff = elapsed_days.days
    return Series_days_diff


def CalcLoginRate(df_dau, df_install):

    date_conv_func = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')

    df_dau_inst = pd.merge(df_dau, df_install,
             on='user_id', suffixes=('', '_inst'))
    df_dau_inst['log_date'] = df_dau_inst['log_date'].apply(date_conv_func)
    df_dau_inst['log_date_inst'] = df_dau_inst['log_date_inst'].apply(date_conv_func)
    df_dau_inst.is_copy = False
    df_dau_inst['elapsed_days'] = df_dau_inst.apply(lambda x: (x.log_date - x.log_date_inst), axis=1)
    df_dau_inst['elapsed_days'] =\
        df_dau_inst['elapsed_days'].apply(
            lambda x: x.astype('timedelta64[D]') / np.timedelta64(1, 'D'))
    df_dau_inst_7_13 = df_dau_inst[(df_dau_inst['elapsed_days'] == 7) &
                                   (df_dau_inst['elapsed_days'] <= 13)]

    density_func = lambda x: float(len(x)) / 7
    df_dau_inst_7_13_login_ds =\
        df_dau_inst_7_13.groupby('user_id').apply(density_func)
    df_dau_inst_7_13_login_ds = pd.DataFrame(
        {'user_id': df_dau_inst_7_13['user_id'],
         'density': df_dau_inst_7_13_login_ds})

    return df_dau_inst_7_13_login_ds


def CallSrtToTime(str_date_format):
    var_datetime = datetime.datetime.strptime(str_date_format, '%Y-%m-%d')

    return var_datetime


def MergeTargetUserAndLoginRate(df_install, df_dau_inst_7_13_login_ds):
    date_conv_func = lambda x: datetime.datetime.strptime(x, '%Y-%m-%d')
    df_install['log_date'] = df_install['log_date'].apply(date_conv_func)
    df_target_install =\
        df_install[(df_install['log_date'] >= CallSrtToTime('2013-06-01')) &
                   (df_install['log_date'] <= CallSrtToTime('2013-08-25'))]
    print(df_install.columns)
    print(df_dau_inst_7_13_login_ds.head().columns)
    df_target_install_login_ds = pd.merge(df_target_install,
                                          df_dau_inst_7_13_login_ds,
                                          on='user_id',
                                          how='left')
    df_target_install_login_ds.fillna(0)

    return df_target_install_login_ds


def PrepareBattleData(df_battle, df_install):

    df_battle_inst = pd.merge(df_battle, df_install,
                              on='user_id',
                              suffixes=('', '_inst'))
    df_battle_inst['elapsed_days'] = df_battle_inst.apply(GetDaysDiff, axis=1)
    df_battle_inst2 = df_battle_inst[(df_battle_inst['elapsed_days'] >= 0) &
                               (df_battle_inst['elapsed_days'] <= 6)]

    AddPrefixD = lambda x: 'd{}'.format(x)
    df_battle_inst2['elapsed_days'] = df_battle_inst2['elapsed_days'].apply(AddPrefixD)
    battle_inst2_cast = df_battle_inst2.pivot_table(values='count',
                                                 index='user_id',
                                                 columns='elapsed_days',
                                                 fill_value=0,
                                                 aggfunc=np.sum)
    # print battle_inst2_cast.head()

    return battle_inst2_cast


def DoPCA(df_input, N_DIM):
    array_input = df_input.values
    pca = PCA(n_components=N_DIM)
    pca.fit(array_input)
    X_pca = pca.transform(array_input)

    df_pca = pd.DataFrame(X_pca)

    return df_pca


def ExecutePCAProcess(df_battle_inst2_cast, DIM_NUM=3):
    # PCAかける前に正規化しておく
    battle_inst2_cast_prop = df_battle_inst2_cast
    CalcRatio = lambda x: x.values / float(np.sum(x))
    battle_inst2_cast_prop = battle_inst2_cast_prop.apply(CalcRatio, axis=1)

    print(u'Input ShapeDF is {}'.format(battle_inst2_cast_prop.shape))
    array_after_pca = DoPCA(df_battle_inst2_cast, 7)
    df_battle_inst2_cast_pca = pd.DataFrame(array_after_pca)
    print(u'Out ShapeDF is {}'.format(battle_inst2_cast_prop.shape))
    print(u'-'*30)
    df_battle_inst2_cast_pca['user_id'] = df_battle_inst2_cast.index

    return battle_inst2_cast_prop, df_battle_inst2_cast_pca


def PrepareMessageDate(df_msg, df_install):
    df_msg_inst = pd.merge(df_msg, df_install,
                           on='user_id',
                           suffixes=('', '_inst'))
 
    df_msg_inst['elapsed_days'] = df_msg_inst.apply(GetDaysDiff, axis=1)
    df_msg_inst2 = df_msg_inst[(df_msg_inst['elapsed_days'] >= 0) &
                               (df_msg_inst['elapsed_days'] <= 6)]


def PrepareMergedData(df_input1, df_input2):
    df_merged = pd.merge(df_input1, df_input2,
                           on='user_id',
                           suffixes=('', '_inst'))
 
    df_merged['elapsed_days'] = df_merged.apply(GetDaysDiff, axis=1)
    df_merged = df_merged[(df_merged['elapsed_days'] >= 0) &
                               (df_merged['elapsed_days'] <= 6)]
    
    AddPrefixD = lambda x: 'd{}'.format(x)
    df_merged['elapsed_days'] = df_merged['elapsed_days'].apply(AddPrefixD)
    df_merged_cast = df_merged.pivot_table(values='count',
                                           index='user_id',
                                           columns='elapsed_days',
                                           fill_value=0,
                                           aggfunc=np.sum)

    return df_merged_cast


def CallKmeans(input_df, NUM_CLUSTERS=10, MAX_ITER=100, N_INIT=10):
    km = cluster.KMeans(n_clusters=NUM_CLUSTERS,
                init='k-means++', n_init=N_INIT,
                verbose=True, max_iter=MAX_ITER)  # Kmeansインスタンスを作成
    km.fit(input_df)  # 実データにfitting
    labels = km.labels_  # 各要素にクラス番号をふる  戻り値はnumpy.ndarray

    return labels


def DoKmeansProcess(column_prefix_name, df_freq, df_prob, df_pca):
    """
    kmenasを使ったクラスタリング処理（と前・後処理）を実施する
    ID3からID4のみでforループ回して、kmeansの実行。ラベルを取得する。
    """
    # for dev
    column_prefix_name = "battle"
    df_freq = battle_install_merged_cast
    df_prob = battle_install_prob
    df_pca = battle_install_pca
    column_id = 3
    # params
    ID_RANGE_FROM = 3
    ID_RANGE_TO = 6


    # スタック用のデータフレーム
    df_stack['user_id'] = pd.DataFrame(df_freq.index)

    for column_id in range(ID_RANGE_FROM, ID_RANGE_TO + 1):
        freq_col_name = 'cluster_{}_freq_{}'.format(column_prefix_name, column_id)
        df_stack[freq_col_name] = CallKmeans(df_freq, column_id)
        prob_col_name = 'cluster_{}_prob_{}'.format(column_prefix_name, column_id)
        df_stack[prob_col_name] = CallKmeans(df_prob, column_id)
        pca_col_name = 'cluster_{}_pca_{}'.format(column_prefix_name, column_id)
        df_stack[pca_col_name] = CallKmeans(df_pca, column_id)





def PrepareEachData(df_input1, df_input2, DIM_NUM=7):
    df_merged_cast = PrepareMergedData(df_input1, df_input2)  # 前処理
    df_cast_prop, df_cast_pca =\
        ExecutePCAProcess(df_merged_cast, DIM_NUM)  # PCA

    return df_merged_cast, df_cast_prop, df_cast_pca


def Main():
    df_install = ReadInstall("game-01", "2013-09-30")

    df_dau = ReadSeqDate("game-01", "2013-06-01", "2013-09-30", "dau",
                         "../sample_data/sample-data/section9/daily/dau")

    df_battle = ReadSeqDate("game-01", "2013-06-01", "2013-08-31", "action",
                            "../sample_data/sample-data/section9/daily/action/",
                            "battle")

    df_msg = ReadSeqDate("game-01", "2013-06-01", "2013-08-31", "action",
                            "../sample_data/sample-data/section9/daily/action/",
                            "message")

    df_help = ReadSeqDate("game-01", "2013-06-01", "2013-08-31", "action",
                            "../sample_data/sample-data/section9/daily/action/",
                            "help")

    df_dau_inst_7_13_login_ds = CalcLoginRate(df_dau, df_install)
    df_target_install_login_ds = MergeTargetUserAndLoginRate(df_install, df_dau_inst_7_13_login_ds)
    # バトルとインストールからなるデータの用意
    battle_install_merged_cast, battle_install_prob, battle_install_pca = PrepareEachData(df_battle, df_install)
    # メッセージとインストールからなるデータの用意
    msg_install_merged_cast, msg_install_prob, msg_install_pca = PrepareEachData(df_msg, df_install)
    # 協力とインストールからなるデータの用意
    help_install_merged_cast, help_install_prob, help_install_pca = PrepareEachData(df_help, df_install)

    # kmeansクラスタリングする関数を作成
    # 引数にとるのは、列の名前、データ、確率化したデータ、pcaかけたデータの３種
    # RETURN user_idが１列とデータ種別が３*ID3~6へのクラスタ番号　の合計 １３列のはず

    # バトル、メッセージ、協力　の順にkmenasクラスタリングを実施する


if __name__ == '__main__':
    Main()
