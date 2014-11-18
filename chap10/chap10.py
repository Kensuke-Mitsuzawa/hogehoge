#! /usr/bin/python
# -*- coding:utf-8 -*-
__author__ = 'kensuke-mi'

import pandas as pd
import datetime
import numpy as np
import os
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
import multiprocessing
n_multi_core = multiprocessing.cpu_count()


def read_action_hourly(app_name, date_from, date_to):
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

    data_path_base = '../sample_data/sample-data/section10/action_hourly'
    f_name = 'action_hourly.tsv'

    x = pd.DataFrame()
    for date in date_seq:
        data_path = os.path.join(data_path_base, app_name, date, f_name)
        loaded_data = pd.io.parsers.read_csv(
            filepath_or_buffer=data_path, sep="\t")
        x = x.append(loaded_data)

    return x

def RandomTest():
    from sklearn.ensemble import RandomForestClassifier

    trainingdata = [[1, 1], [2, 2], [-1, -1], [-2, -2]]
    traininglabel = [1, 1, -1, -1]
    testdata = [[3, 3], [-3, -3]]

    model = RandomForestClassifier()
    model.fit(trainingdata, traininglabel)
    output = model.predict(testdata)

    for label in output: print label


def preprocess_action_log(action_hourly):
    """
    training dataとanswer dataをデータフレームで用意する
    :param action_hourly:
    :return: tuple()
    """
    # 時間帯が列になるように整形する
    df_stack = []
    dates = list(action_hourly.log_date.unique())
    dates_for_train = dates[:-1]
    ifelse_func = lambda x: 1 if sum(x) > 7 else 0
    for day_index, day in enumerate(dates_for_train):
        x = action_hourly[action_hourly.log_date == day]
        # 日毎に、index=user_id, column=hour, value=0,1のdfを作る
        x_pivot = pd.pivot_table(x, index='user_id', columns='log_hour', values='count', fill_value=0, aggfunc=ifelse_func).reset_index()
        col_names_list = ['p{}_{}'.format(day_index + 1, hour) for hour in range(0, 24)]
        x_pivot.columns = ['user_id'] + col_names_list
        df_stack.append(x_pivot)

    # 説明変数用のデータを用意する
    df_train_data = df_stack[0]
    for list_index, day_user_df in enumerate(df_stack):
        if list_index == 0: continue
        df_train_data = pd.merge(df_train_data, df_stack[list_index], on='user_id', how='left')
        df_train_data = df_train_data.fillna(0)

    # 目的変数のデータを用意する
    ans_data_rows = action_hourly[action_hourly.log_date == dates[-1]]
    ans_data = pd.pivot_table(ans_data_rows, index='user_id', columns='log_hour', values='count', fill_value=0, aggfunc=ifelse_func).reset_index()
    col_names_list = ['a_{}'.format(hour) for hour in range(0, 24)]
    ans_data.columns = ['user_id'] + col_names_list

    # 訓練用と答えのデータをuser_idでmergeしておく
    df_preprocessed = pd.merge(df_train_data, ans_data, on='user_id', how='inner')

    return df_preprocessed


def prepare_array_like_data(df_preprocessed):
    """
    scikit-learnのinputに適した形でのデータの作成をする
    作成すべきデータは、１時間毎のユーザーログインするか、しないか、の対応関係のデータ

    ０時のログインデータ（月曜から〜日曜）が訓練で、次の週のある日のログインデータが答え
    training
    ndarray:
    [[user_1_monday, user_1_tuesday, ... user_1_sunday],
    [user_2_monday, user_2_tuesday, ... user_2_sunday],
    ...
    [user_n_monday, user_n_tuesday, ... user_n_sunday]]

    gold:
    [login_user_1, login_user_2, ..., login_user_n]

    :return: Map {key int hour: value tuple train_ans_tuple (train_2d_array, ans_1d_array)}
    """
    stack_map = {}

    for target_hour in range(0, 24):
        list_target_cols_training = [
            u'p{}_{}'.format(target_date, target_hour) for target_date in range(1, 8)]
        # list_target_cols = list_target_cols_training + ['a_{}'.format(target_hour)]
        array_training_data = df_preprocessed.ix[:, list_target_cols_training].values
        array_ans_data = df_preprocessed.ix[:, ['a_{}'.format(target_hour)]].values
        # 2d arrayはwarning出るので、1d arrayにする
        array_ans_data_1d = np.transpose(array_ans_data)[0, :]

        stack_map[target_hour] = (array_training_data, array_ans_data_1d)

    return stack_map


def dimention_preprocess(row_array_data, mean=True, std=True):
    """

    :param row_array_data:
    :param mean: 平均化を行う
    :param std: 偏差揃えを行う
    :return:
    """
    scaler = preprocessing.StandardScaler(with_mean=mean,
                                          with_std=std).fit(row_array_data)
    normalized_array = scaler.transform(row_array_data)

    return normalized_array


def call_random_forest(input_training, input_gold, Grid=False):
    """

    :param input_training:
    :param input_gold:
    :param Grid:
    :return:
    """
    if Grid == True:
        #tuned_parameters = [{'n_estimators': [10, 30, 50, 70, 90, 110, 130, 150],
        #                     'max_features': ['auto', 'sqrt', 'log2', None]}]
        tuned_parameters = [{'n_estimators': [10],
                             'max_features': ['auto']}]
        clf = GridSearchCV(RandomForestClassifier(),
                           tuned_parameters, cv=2, scoring='accuracy', n_jobs=n_multi_core)
        res = clf.fit(input_training, input_gold)

        result_map = {'model': clf, 'res': res, 'gold': input_gold}

    else:
        # アンサンブル学習を実行する
        rf = RandomForestClassifier()
        rf.fit(input_training, input_gold)
        res = rf.predict(input_training)

        result_map = {'model': rf, 'res': res, 'gold': input_gold}

    return  result_map


def __UNUSED_prepare_array_like_data(action_hourly):
    """
    これだと、user_idで対応関係が作れないことがわかったので、この関数は使わない

    scikit-learnのinputに適した形でのデータの作成をする
    作成すべきデータは、１時間毎のユーザーログインするか、しないか、の対応関係のデータ

    ０時のログインデータ（月曜から〜日曜）が訓練で、次の週のある日のログインデータが答え
    training
    ndarray:
    [[user_1_monday, user_1_tuesday, ... user_1_sunday],
    [user_2_monday, user_2_tuesday, ... user_2_sunday],
    ...
    [user_n_monday, user_n_tuesday, ... user_n_sunday]]

    gold:
    [login_user_1, login_user_2, ..., login_user_n]

    :return: Map {key int hour: value tuple train_ans_tuple (train_2d_array, ans_1d_array)}
    """
    dates = list(action_hourly.log_date.unique())
    dates_for_train = dates[:-1]
    date_for_gold = dates[-1]
    ifelse_func = lambda x: 1 if sum(x) > 7 else 0
    # 訓練データの日付けだけを選択
    df_for_train = action_hourly[action_hourly['log_date'].str.contains('|'.join(dates_for_train))]
    df_for_gold = action_hourly[action_hourly['log_date'].str.contains(date_for_gold)]

    stack_map = {}
    # 時間ごとに訓練データとゴールドデータの作成
    for target_hour in range(0, 24):

        df_train_for_target_hour = df_for_train[df_for_train.log_hour == target_hour]
        df_gold_for_target_hour = df_for_gold[df_for_gold.log_hour == target_hour]

        # 0,1のデータに変換する
        training_data = pd.pivot_table(df_train_for_target_hour,
                                       index='user_id',
                                       columns='log_date',
                                       values='count',
                                       fill_value=0, aggfunc=ifelse_func).reset_index()
        array_training_data = training_data.drop(u'user_id', axis=1).values

        gold_data = pd.pivot_table(df_gold_for_target_hour,
                                       index='user_id',
                                       columns='log_date',
                                       values='count',
                                       fill_value=0, aggfunc=ifelse_func).reset_index()
        array_gold_data = gold_data.drop(u'user_id', axis=1).values
        array_gold_data_1d = np.transpose(array_gold_data)[0, :]

        # user_idとarray_training_dataとarray_gold_data_1dをmapに記録する
        stack_map[target_hour] = (user_id, array_training_data, array_gold_data_1d)


def Main():
    action_hourly = read_action_hourly('game-01', '2013-08-01', '2013-08-08')
    df_preprocessed = preprocess_action_log(action_hourly)
    input_stack_map = prepare_array_like_data(df_preprocessed)

    list_result_stack = []
    for target_hour in input_stack_map:
        input_tuple = input_stack_map[target_hour]
        scaled_input_array = dimention_preprocess(input_tuple[0])
        result_map = call_random_forest(scaled_input_array, input_tuple[1], Grid=False)
        result_map['hour'] = target_hour
        list_result_stack.append(result_map)

    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import accuracy_score
    # モデルを２４個用意して、各モデルの予測精度を確認する
    for model_result_map in list_result_stack:
        # confusion matrixを作成してモデルの評価をしてみる
        #print result_map['res']
        #print result_map['gold']
        print 'Summary for Model hour:{}'.format(model_result_map['hour'])
        print '='*30
        print 'Confusion Matrix'
        print confusion_matrix(model_result_map['res'], model_result_map['gold'])  # 引数に結果とgold
        print '- - - '*6
        print 'model accuracy'
        print accuracy_score(model_result_map['res'], model_result_map['gold'])  # accucyを求める
        print '='*30


if __name__ == '__main__':
    Main()



