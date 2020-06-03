# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np

def catch_score(x, group):
    for i in group:
        if x > float(eval(i)[0]) and x <= float(eval(i)[1]):
            bin = i
    try:
        return bin
    except:
        print(x, eval(i)[0])
        bin = 0
        return bin
def func_good(x):
    if x == 1:
        y = 0
    else:
        y = 1
    return y


def func_bad(x):
    if x == 0:
        y = 0
    else:
        y = 1
    return y

def group_score(data, code, cut, score_name=None):
    '''
    :param data:
    :param code:
    :param cut: qcut/cut/10score_cut/uncut
    :param score_name: 可选填，当score的名字自定义非'score'时
    :return:
    '''
    df_rs = data.copy()
    list = df_rs.index.tolist()
    df_rs.ix[:, 'no'] = [ii for ii in range(len(list))]
    if score_name:
        df_rs['score'] = df_rs[score_name]
    try:
        if cut == 'qcut':
            a = pd.qcut(df_rs['score'], 10)
            df_rs['score_group'] = a
        elif cut == 'cut':
            a = pd.cut(df_rs['score'], 10)
            df_rs['score_group'] = a
        elif cut == 'uncut':
            df_rs['score_group'] = df_rs['score'][:]
        elif cut == '10score_cut':
            group = ['[' + str(i) + ',' + str(i + 10) + ']' for i in range(100, 1300, 10)]
            df_rs['score_group'] = df_rs['score'].map(lambda x: catch_score(x, group))
        elif cut == '2score_cut':
            group = ['[' + str(i) + ',' + str(i + 10) + ']' for i in range(100, 1300, 2)]
            df_rs['score_group'] = df_rs['score'].map(lambda x: catch_score(x, group))
        elif cut == 'artificial':
            df_rs['score_group'] = df_rs['score_bin']
        else:
            print('error! check cut_value，cut_method changed to uncut  ')
            df_rs['score_group'] = df_rs['score'][:]
    except:
        print('error! check cut_value，cut_method changed to uncut  ')
        df_rs['score_group'] = df_rs['score'][:]
    df_rs['good'] = df_rs[code].map(lambda x: func_good(x))
    df_rs['bad'] = df_rs[code].map(lambda x: func_bad(x))
    b = df_rs.groupby(['score_group', ])['no', 'good', 'bad'].sum()
    df_gp = pd.DataFrame(b)
    # print df_gp
    return df_gp

def gb_add_woe(data, code, cut, score_name=None):
    if score_name:
        df_gp = group_score(data, code, cut, score_name)
    else:
        df_gp = group_score(data, code, cut)
    df_gp.ix[:, 'bad_rate'] = df_gp['bad'] / (df_gp['bad'] + df_gp['good'])
    df_gp['total_cnt'] = df_gp['bad'] + df_gp['good']
    bad_sum = df_gp['bad'].sum()
    good_sum = df_gp['good'].sum()
    # df_gp['bad_pct_alone']=df_gp['bad']/(df_gp['bad']+df_gp['good'])
    df_gp['bad_pct'] = df_gp['bad'].apply(lambda x: x / float(bad_sum))
    df_gp['good_pct'] = df_gp['good'].apply(lambda x: x / float(good_sum))
    # df_gp['odds'] = df_gp.apply(lambda x: x['good_pct'] - x['bad_pct'])
    df_gp['odds'] = df_gp['good_pct'] / df_gp['bad_pct']
    df_gp.ix[:, 'Woe'] = np.log(df_gp['good_pct'] / df_gp['bad_pct'])
    df_gp.ix[:, 'IV'] = (df_gp['good_pct'] - df_gp['bad_pct']) * df_gp['Woe']
    bad_cnt = df_gp['bad'].cumsum()
    good_cnt = df_gp['good'].cumsum()
    df_gp.ix[:, 'bad_cnt'] = bad_cnt
    df_gp.ix[:, 'good_cnt'] = good_cnt
    df_gp.ix[:, 'b_c_p'] = df_gp['bad_cnt'] / np.array([bad_sum for _ in range(len(df_gp))])
    df_gp.ix[:, 'g_c_p'] = df_gp['good_cnt'] / np.array([good_sum for _ in range(len(df_gp))])
    df_gp.ix[:, 'KS'] = df_gp['b_c_p'] - df_gp['g_c_p']
    df_gp['bin'] = df_gp.index.map(lambda x: str(x)).tolist()[:]
    # df_gp['bin'] = df_gp.index.map(lambda x: str(x) + ')').tolist()[:]
    df_gp['bin_pct'] = (df_gp['good'] + df_gp['bad']) / (bad_sum + good_sum)
    df_gp = df_gp.reset_index(drop=True)
    ks_max = df_gp['KS'].max()
    df_gp = df_gp[['bin', 'total_cnt', 'good', 'bad', 'bin_pct', 'bad_rate', 'Woe', 'KS', 'IV']]
    # print df_gp
    return ks_max, df_gp

def auc(df, label='label', predict='predict', prob=False):
    """
    功能：根据样本实际标签和预测概率（分数）快速计算auc值。
    参数：
        df: pd.DataFrame,至少包含标签号预测结果列；
        label: 样本实际标签(0, 1)；
        predict: 预测结果（分数或概率均可）；
        prob: predict是否为概率，根据实际情况设置，默认False；
    输出：auc值。
    """
    if prob:
        df.sort_values(by=predict, ascending=False, inplace=True)
    else:
        df.sort_values(by=predict, ascending=True, inplace=True)
    rank = list(reversed(range(1, df.shape[0] + 1)))
    df['rank'] = rank
    mean = df.groupby([predict])['rank'].mean().reset_index()
    mean.columns = [predict, 'rank_mean']
    df = pd.merge(df, mean, on=predict)
    print(df)
    df[label].value_counts()
    N, M = df[label].value_counts().sort_index().values
    print(df[label].value_counts().sort_index())
    formula1 = df[df[label]==1]['rank'].sum()
    formula2 = M * (M + 1) / 2
    return np.round((formula1 - formula2) / (M * N), 4)

def ks_auc():
    df_rs_score=pd.read_csv('./new_data.csv')
    time_cut=[ 'd01:1-14','d02:15-30', 'd03:31-60']
    time_name=['近14天','近30天','近60天']
    ks_value=[]
    AUC1=[]
    AUC2=[]
    from sklearn.metrics import roc_auc_score
    for ii in range(3):
        new_l=time_cut[:ii+1]
        df_rs_score_ii=df_rs_score[df_rs_score['gap_days'].isin(new_l)]
        print(df_rs_score_ii.shape)
        ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score_ii, 'credit_status', 'uncut',score_name='nydd_prea_score')
        auc1=auc(df_rs_score_ii,label='credit_status', predict='nydd_prea_score')
        Y=df_rs_score_ii['credit_status']
        p=1-df_rs_score_ii['nydd_prea_p']
        auc2 = roc_auc_score(Y, p)
        print(ks_max_uncut)
        ks_value.append(ks_max_uncut)
        AUC1.append(auc1)
        AUC2.append(auc2)
    df=pd.DataFrame([time_name,ks_value,AUC1,AUC2])
    print(df)
    df.to_csv('result.csv',index=False)

ks_auc()
