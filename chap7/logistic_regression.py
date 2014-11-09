#! /usr/bin/python
# -*- coding:utf-8 -*-
import pandas as pd
import statsmodels.api as sm


def LoadData():
    DATA_PATH = '../sample_data/section7-dau.csv'
    dau = pd.io.parsers.read_csv(
        filepath_or_buffer=DATA_PATH, sep=",")
    dau.head()

    return dau


def UniqueData(data_set):
    mau = data_set.drop_duplicates(
        data_set.ix[:,
                    ['region_month', 'device', 'user_id']])

    sp_mau = data_set.ix[
        data_set.device == 'SP',
        ['region_month', 'device', 'user_id']]
    # pandasではunique()関数の代わりにdrop_duplicate()メソッド使う
    sp_mau = sp_mau.drop_duplicates()

    fp_mau = data_set.ix[
        data_set.device == 'FP',
        ['region_month', 'device', 'user_id']]
    fp_mau = fp_mau.drop_duplicates()

    return mau, sp_mau, fp_mau


def SplitData(mau, fp_mau, sp_mau):
    fp_mau1 = fp_mau.ix[fp_mau.region_month == '2013-01', :]
    fp_mau2 = fp_mau.ix[fp_mau.region_month == '2013-02', :]

    sp_mau1 = sp_mau.ix[sp_mau.region_month == '2013-01', :]
    sp_mau2 = sp_mau.ix[sp_mau.region_month == '2013-02', :]

    return fp_mau1, fp_mau2, sp_mau1, sp_mau2


def MergeData(mau, fp_mau1, fp_mau2, sp_mau1, sp_mau2):
    # 列の追加操作
    mau.is_copy = False
    mau['is_access'] = 1

    fp_mau1 = pd.merge(fp_mau1,
                       mau.ix[mau.region_month == '2013-02',
                              ['user_id', 'is_access']],
                       on='user_id',
                       how='left')

    # FP ２月のデータの用意
    fp_mau2.is_copy = False
    fp_mau2['is_fp'] = 1
    fp_mau1 = pd.merge(
        fp_mau1,
        fp_mau2.ix[:, ['user_id', 'is_fp']],
        on='user_id',
        how='left')
    fp_mau1 = fp_mau1.fillna(0)  # NaNが出ている部分を0で置換する

    # １月は携帯電話からの利用、２月はスマホからの利用ユーザーの区別
    sp_mau2.is_copy = False
    sp_mau2['is_sp'] = 1
    fp_mau1 = pd.merge(fp_mau1,
                       sp_mau2.ix[:, ['user_id', 'is_sp']],
                       on='user_id',
                       how='left')
    fp_mau1 = fp_mau1.fillna(0)

    # fp_mauから２月の利用がないユーザー、または２月がスマホのユーザーを抽出
    # 複数の条件指定の際には（）表記が必要なので、忘れないように
    fp_mau1 = fp_mau1[(fp_mau1.is_access == 0) | (fp_mau1.is_sp == 1)]

    return fp_mau1


def PrepareAccessLogPerData(dau, fp_mau1):
    fp_dau = dau[(dau.device == 'FP') & (dau.region_month == '2013-01')]
    fp_dau.is_copy = False
    fp_dau['is_access'] = 1

    # reshap2::dcastに相当する処理
    # user_idはインデックスに成る点に注意
    fp_dau1_cast = fp_dau.pivot_table(values='is_access',
                                      index='user_id',
                                      columns='region_day',
                                      fill_value=0)
    # 列名の変更
    list_day_names = ['X{}day'.format(day) for day in range(1, 32)]
    fp_dau1_cast.columns = (list_day_names)
    # user_idはインデックスになっているので、新しい列にしておく
    fp_dau1_cast['user_id'] = fp_dau1_cast.index

    # ２月利用でかつスマホからの利用者をくっつける
    fp_dau1_cast = pd.merge(fp_dau1_cast,
                            fp_mau1.ix[:, ['user_id', 'is_sp']],
                            on='user_id',
                            how='inner')

    print len(fp_dau1_cast[fp_dau1_cast.is_sp == 0])
    print len(fp_dau1_cast[fp_dau1_cast.is_sp == 1])

    return fp_dau1_cast


def CallLogisticRegression(fp_dau1_cast):
    fp_dau1_cast = fp_dau1_cast.drop('user_id', 1)
    list_explain_variable = ['X{}day'.format(day) for day in range(1, 32)]
    X = fp_dau1_cast.ix[:, list_explain_variable]
    X = sm.add_constant(X, prepend=False)
    Y = fp_dau1_cast['is_sp']
    model = sm.GLM(Y, X)
    results = model.fit()
    print results.summary()

if __name__ == '__main__':
    dau = LoadData()
    mau, sp_mau, fp_mau = UniqueData(dau)
    fp_mau1, fp_mau2, sp_mau1, sp_mau2 = SplitData(mau, fp_mau, sp_mau)
    fp_mau1 = MergeData(mau, fp_mau1, fp_mau2, sp_mau1, sp_mau2)
    fp_dau1_cast = PrepareAccessLogPerData(dau, fp_mau1)
    """
    # TODO
    # bionomialのオプションを与える
    # stepに相当するものをみつける 
    # モデルのサマリを見れるようにしておく。特にAIC
    """
    CallLogisticRegression(fp_dau1_cast)
