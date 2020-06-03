# -*- coding:utf-8 -*-


import numpy as np
import pandas as pd
import tool
from numpy import array


def gb_add_woe(data):
    df_gp = data.copy()
    df_gp['pct_default'] = df_gp['bad'] / (df_gp['bad'] + df_gp['good'])
    df_gp = df_gp.sort_values(by=['pct_default'], ascending=False)
    bad_sum = df_gp['bad'].sum()
    good_sum = df_gp['good'].sum()
    df_gp['bad_pct'] = df_gp['bad'] / bad_sum
    df_gp['good_pct'] = df_gp['good'] / good_sum
    df_gp['odds'] = df_gp['good_pct'] - df_gp['bad_pct']
    df_gp['Woe'] = np.log(df_gp['good_pct'] / df_gp['bad_pct'])
    df_gp['IV'] = (df_gp['good_pct'] - df_gp['bad_pct']) * df_gp['Woe']
    df_gp['bad_cnt'] = df_gp['bad'].cumsum()
    df_gp['good_cnt'] = df_gp['good'].cumsum()
    df_gp['b_c_p'] = df_gp['bad_cnt'] / bad_sum
    df_gp['g_c_p'] = df_gp['good_cnt'] / good_sum
    df_gp['KS'] = abs(df_gp['g_c_p'] - df_gp['b_c_p'])
    ks_max = df_gp['KS'].max()
    return ks_max, df_gp


def set_tag(x):
    if x == 1:
        y = 1
        z = 0
    else:
        y = 0
        z = 1
    return y, z


def cut_method(data1, factor_name, flag_name, method, n):
    data = data1.copy()
    if method == 'qcut':
        data[factor_name] = pd.qcut(data[factor_name], n, duplicates='drop')
    elif method == 'cut':
        data[factor_name] = pd.cut(data[factor_name], n)
    elif method == 'uncut':
        data[factor_name] = data[factor_name][:]
    elif method == 'cumsum':
        values = pd.DataFrame(data[factor_name].value_counts()).sort_index()
        values['cum'] = values[factor_name].cumsum()
        # sort_values=np.sort(data[factor_name].value_counts().index)
        sum = data[factor_name].shape[0]
        sd_bin = sum / n
        botton = float(list(values.index)[0]) - 0.001
        values_c = values.copy()
        bin = []
        for i in range(n):
            values_i = values_c[values_c['cum'] <= sd_bin]
            if values_i.shape[0] == 0:
                top = list(values_c.index)[0]
            else:
                top = list(values_i.index)[-1]
            bin.append([botton, top])
            botton = top
            values_c = values_c[values_c.index > top]

            values_c['cum'] = values_c[factor_name].cumsum()
            if values_c.shape[0] == 0:
                break
        # bin=[]
        # for i in sort_values:
        #     data_i=data[(data[factor_name]>botton)&(data[factor_name]<=i)]
        #     if data_i.shape[0]>sd_bin:
        #         bin.append([botton,i])
        #         botton=i
        bin.append([botton, list(values.index)[-1]])
        data[factor_name] = data[factor_name].map(lambda x: find_bin(x, bin))
    return data


def find_bin(x, list_bin):
    for i in list_bin:
        if x > i[0] and x <= i[1]:
            y = str(i)

    try:
        return y
    except:
        print(x, list_bin)


def loop(df, factor_name, flag_name, method):
    '''
    用于寻找保证单调性下的最大分Bin组数
    :param df:
    :param factor_name:
    :param ex_value:
    :return:
    '''
    find_n = []
    data_for_cut = df.copy()
    for n in range(2, 10, 1):
        print('loop to ', n)
        # try:
        data_1 = cut_method(data_for_cut, factor_name, flag_name, method, n)
        data_gp = data_1.groupby(factor_name).sum()
        data_gp['sort'] = data_gp.index.map(lambda x: float(x.split(',')[0][1:]))
        df_jugde = data_gp.sort_values('sort').drop('sort', axis=1)
        # pct_list=gb_add_woe(df_jugde)[1]['pct_default'].tolist()
        pct_list = list(df_jugde['bad'] / (df_jugde['bad'] + df_jugde['good']))
        if pd.Series(pct_list).is_monotonic_decreasing or pd.Series(pct_list).is_monotonic_increasing:
            find_n.append(n)
        else:
            break
            # except Exception,e:
            #     print Exception,e
            #     print 'cant cut more'
            #     break
    # print find_n
    max_n = max(find_n)
    return max_n


def format_dtype(x,var_type):
    try:
       if pd.isnull(x) or x.find('NAN')>=0:
           return (x)
       elif var_type=='str':
           return str(x)
       else:
           return float(x)
    except:
        print('error',x)
        return 'error'

def cut_cumsum_bin(data, factor_name, flag_name, not_in_list, method, mono=True, bin_num=20,var_type=''):
    data = data[[factor_name, flag_name]].fillna('NAN')
    data['bad'] = data[flag_name].map(lambda x: set_tag(x)[0])
    data['good'] = data[flag_name].map(lambda x: set_tag(x)[1])
    df_ex1 = data[data[factor_name].map(lambda x: str(x) in not_in_list)]  # 分割需要单独切出的bin值(-1，nan),单独值的bin使用[a,a]来表示
    df_ex = df_ex1.copy()
    df_ex[factor_name] = df_ex[factor_name].map(lambda x: '(' + str(x) + ',' + str(x) + ')')
    df_rm1 = data[data[factor_name].map(lambda x: str(x) not in not_in_list)]
    ##
    if mono:
        n = loop(df_rm1, factor_name, flag_name, method)
    else:
        n = min(bin_num, len(set(data[factor_name])))
    # print 'cut',n
    df_rm = df_rm1.copy()
    df_rm[factor_name] = cut_method(df_rm, factor_name, flag_name, method, n)
    # print df_rm
    df = pd.concat([df_ex, df_rm], axis=0)
    df_ = df.groupby(factor_name).sum()[['bad', 'good']]
    df_gp = gb_add_woe(df_)[1]
    df_gp[factor_name] = df_gp.index
    # df_gp['sort'] = df_gp[factor_name].map(lambda x: float(str(x).split(',')[0][1:]))
    df_gp = df_gp.sort_values('pct_default')
    df_new = df_gp.copy()
    df_new['Total_Num'] = df_new['bad'] + df_new['good']
    df_new['Bad_Num'] = df_new['bad']
    df_new['Good_Num'] = df_new['good']
    df_new['Total_Pcnt'] = df_new['Total_Num'] / df_new['Total_Num'].sum()
    df_new['Bad_Rate'] = df_new['Bad_Num'] / df_new['Total_Num']
    df_new['Good_Rate'] = df_new['Good_Num'] / df_new['Total_Num']
    df_new['Good_Pcnt'] = df_new['Good_Num'] / df_new['Good_Num'].sum()
    df_new['Bad_Pcnt'] = df_new['Bad_Num'] / df_new['Bad_Num'].sum()
    df_new['Bad_Cum'] = df_new['b_c_p']
    df_new['Good_Cum'] = df_new['g_c_p']
    df_new['Gini'] = 1 - pow(array(list(df_new['Good_Rate'])), 2) - pow(array(list(df_new['Bad_Rate'])), 2)
    df_new = df_new[
        [factor_name, 'Total_Num', 'Bad_Num', 'Good_Num', 'Total_Pcnt', 'Bad_Rate', 'Good_Rate', 'Good_Pcnt',
         'Bad_Pcnt', 'Bad_Cum', 'Good_Cum', 'Woe', 'IV', 'Gini', 'KS']]

    if method == 'uncut':
        df_new.columns = [factor_name + '_tmp', 'Total_Num', 'Bad_Num', 'Good_Num', 'Total_Pcnt', 'Bad_Rate',
                          'Good_Rate',
                          'Good_Pcnt',
                          'Bad_Pcnt', 'Bad_Cum', 'Good_Cum', 'Woe', 'IV', 'Gini', 'KS']
        df_new[factor_name] = df_new[factor_name + '_tmp'].apply(
            lambda x: str(x) if str(x).find('NAN') >= 0 else '[' + str(x) + ',' + str(x) + ']')
        df_new_p1=df_new[df_new[factor_name + '_tmp'].apply(lambda x: str(x).find('NAN')>=0)]
        df_new_p2=df_new[df_new[factor_name + '_tmp'].apply(lambda x: str(x).find('NAN')<0)]
        df_new_p2 = df_new_p2.sort_values(factor_name + '_tmp', ascending=True)
        df_new=pd.concat([df_new_p1,df_new_p2])
        df_new = df_new[
            [factor_name, 'Total_Num', 'Bad_Num', 'Good_Num', 'Total_Pcnt', 'Bad_Rate', 'Good_Rate', 'Good_Pcnt',
             'Bad_Pcnt', 'Bad_Cum', 'Good_Cum', 'Woe', 'IV', 'Gini', 'KS']]
        df_new = df_new.reset_index(drop=True)
    else:
        df_new[factor_name] = df_new[factor_name].apply(
            lambda x: '(' + str(x) + ',' + str(x) + ')' if str(x).find('NAN') >= 0 else x)
        df_new = df_new.reset_index(drop=True)
        df_new = df_new.sort_values(factor_name, ascending=True)
    # total_num=sum(df_gp['good'])+sum(df_gp['bad'])
    # df_gp.columns=['Bad_Num','Good_Num','Bad_Rate','Good_Rate','odds','Woe','IV','Bad_cnt','Good_cnt','bcp','gcp','KS','rej_pct','sort','var_name']
    # df_gp['Total_Num']=df_gp['Good_Num']+df_gp['Bad_Num']
    # df_gp['Total_Pcnt']=df_gp['Total_Num']/total_num
    # df_gp[factor_name]=df_gp.index.tolist()

    return df_new
    ##
    # try:
    #     # print 'loop'
    #     if mono:
    #         n = loop(df_rm1, factor_name, flag_name, method)
    #     else:
    #         n = max(20, len(set(data[factor_name])))
    #     # print 'cut',n
    #     df_rm = df_rm1.copy()
    #     df_rm[factor_name] = cut_method(df_rm, factor_name, flag_name, method, n)
    #
    #     # print df_rm
    #     df = pd.concat([df_ex, df_rm], axis=0)
    #     df_ = df.groupby(factor_name).sum()[['bad', 'good']]
    #     df_gp = gb_add_woe(df_)[1]
    #     df_gp[factor_name] = df_gp.index
    #     # df_gp['sort'] = df_gp[factor_name].map(lambda x: float(str(x).split(',')[0][1:]))
    #     df_gp = df_gp.sort_values('pct_default')
    #     df_new = df_gp.copy()
    #     df_new['Total_Num'] = df_new['bad'] + df_new['good']
    #     df_new['Bad_Num'] = df_new['bad']
    #     df_new['Good_Num'] = df_new['good']
    #     df_new['Total_Pcnt'] = df_new['Total_Num'] / df_new['Total_Num'].sum()
    #     df_new['Bad_Rate'] = df_new['Bad_Num'] / df_new['Total_Num']
    #     df_new['Good_Rate'] = df_new['Good_Num'] / df_new['Total_Num']
    #     df_new['Good_Pcnt'] = df_new['Good_Num'] / df_new['Good_Num'].sum()
    #     df_new['Bad_Pcnt'] = df_new['Bad_Num'] / df_new['Bad_Num'].sum()
    #     df_new['Bad_Cum'] = df_new['b_c_p']
    #     df_new['Good_Cum'] = df_new['g_c_p']
    #     df_new['Gini'] = 1 - pow(array(list(df_new['Good_Rate'])), 2) - pow(array(list(df_new['Bad_Rate'])), 2)
    #     df_new = df_new[[factor_name,
    #                      'Total_Num',
    #                      'Bad_Num',
    #                      'Good_Num',
    #                      'Total_Pcnt',
    #                      'Bad_Rate',
    #                      'Good_Rate',
    #                      'Good_Pcnt',
    #                      'Bad_Pcnt',
    #                      'Bad_Cum',
    #                      'Good_Cum',
    #                      'Woe',
    #                      'IV',
    #                      'Gini',
    #                      'KS']]
    #     df_new[factor_name]=df_new[factor_name].apply(lambda x: x if str(x).find('NAN')>=0 else '(' + str(x) + ',' + str(x) +')')
    #     # total_num=sum(df_gp['good'])+sum(df_gp['bad'])
    #     # df_gp.columns=['Bad_Num','Good_Num','Bad_Rate','Good_Rate','odds','Woe','IV','Bad_cnt','Good_cnt','bcp','gcp','KS','rej_pct','sort','var_name']
    #     # df_gp['Total_Num']=df_gp['Good_Num']+df_gp['Bad_Num']
    #     # df_gp['Total_Pcnt']=df_gp['Total_Num']/total_num
    #     # df_gp[factor_name]=df_gp.index.tolist()
    #     df_new = df_new.sort_values('Bad_Rate',ascending=False)
    #     df_new=df_new.reset_index()
    #     return df_new
    # except Exception as e:
    #     print('cannot cut more')
    #     print(Exception, e)
    #     return []


# dic_a=cut_bin_best_ks(rule,['appid','c.create_date','c.end_date','tag3','c.is_overdue','c.overdue_day'],'rule_check','tag3',mono=False)


def multicut_var(data_var):
    pass


# import time
# a=cut_cumsum_bin(data=m_data_train,flag_name='tag7',
#              factor_name='call_in_cell_days_sum_j1m',not_in_list=['NAN','-1.0'],method='cumsum',mono=False)
#


# loop(df,factor_name,flag_name,method)
#
#
# df,factor_name,flag_name,method=test1[['callOut2101TimeMinJ5m','tag']],'callOut2101TimeMinJ5m','tag','cumsum'
# Best_KS_Bin(data=data[[i,'black_flag']],flag_name='black_flag',
#                          factor_name=i,not_in_list=['NaN','-1.0',''])


def chi_square_bin(data, flag_name: str, factor_name: str, not_in_list=['NaN', 'NAN'], mono=False, bin_num=5,
                   var_type='number'):
    from binning.supervised import ChiSquareBinning
    import time
    # X = pd.DataFrame({'a': [random.randint(1, 20) for _ in range(1000)],
    #                   'b': [random.randint(1, 20) for _ in range(1000)],
    #                   'c': [random.randint(1, 20) for _ in range(1000)]})
    # y = [int(random.random() > 0.5) for _ in range(1000)]
    X = data[[factor_name]]
    y = data[flag_name]
    if var_type == 'str':
        categorical_cols = [factor_name]
    else:
        categorical_cols = []
    CB = ChiSquareBinning(max_bin=10, categorical_cols=categorical_cols, force_mix_label=False, force_monotonic=mono,
                          prebin=100, encode=True, strict=False)
    CB.fit(X, y)
    var_bin_list = (CB.bins)[factor_name]
    print(var_bin_list)
    transform_bin = CB.transform(X)
    transform_bin.columns = [ii + '_bin_num' for ii in transform_bin.columns]
    fin = pd.concat([X, transform_bin, y], axis=1)
    var_dict = dict()
    if var_type == 'str':
        dist_num = fin[factor_name + '_bin_num'].unique()
        for ii in dist_num:
            var_str_l = list(set(fin[fin[factor_name + '_bin_num'] == ii][factor_name].tolist()))
            if ii == -1:
                var_str_l = '(NAN,NAN)'
                var_dict[ii] = var_str_l
            else:
                var_dict[ii] = '(%s)' % ','.join(var_str_l)
    else:
        dist_num = fin[factor_name + '_bin_num'].unique()
        if -1 in dist_num:
            var_str_l = '(NAN,NAN)'
            var_dict[-1] = var_str_l
        right_max = var_bin_list[1:]
        bin_list = tool.get_bin_cate(right_max)
        for ii in range(len(bin_list)):
            var_dict[ii + 1] = bin_list[ii]
    fin[factor_name + '_bin'] = fin[factor_name + '_bin_num'].apply(lambda x: var_dict[x])
    new_fin = fin[[factor_name + '_bin', flag_name]]
    new_fin.columns = [factor_name, flag_name]
    new_fin['bad'] = new_fin[flag_name].map(lambda x: set_tag(x)[0])
    new_fin['good'] = new_fin[flag_name].map(lambda x: set_tag(x)[1])
    df_ = new_fin.groupby(factor_name).sum()[['bad', 'good']]
    df_gp = gb_add_woe(df_)[1]
    df_gp[factor_name] = df_gp.index
    # df_gp['sort'] = df_gp[factor_name].map(lambda x: float(str(x).split(',')[0][1:]))
    # df_gp = df_gp.sort_values('pct_default')
    df_new = df_gp.copy()
    df_new['Total_Num'] = df_new['bad'] + df_new['good']
    df_new['Bad_Num'] = df_new['bad']
    df_new['Good_Num'] = df_new['good']
    df_new['Total_Pcnt'] = df_new['Total_Num'] / df_new['Total_Num'].sum()
    df_new['Bad_Rate'] = df_new['Bad_Num'] / df_new['Total_Num']
    df_new['Good_Rate'] = df_new['Good_Num'] / df_new['Total_Num']
    df_new['Good_Pcnt'] = df_new['Good_Num'] / df_new['Good_Num'].sum()
    df_new['Bad_Pcnt'] = df_new['Bad_Num'] / df_new['Bad_Num'].sum()
    df_new['Bad_Cum'] = df_new['b_c_p']
    df_new['Good_Cum'] = df_new['g_c_p']
    df_new['Gini'] = 1 - pow(array(list(df_new['Good_Rate'])), 2) - pow(array(list(df_new['Bad_Rate'])), 2)
    df_new = df_new[
        [factor_name, 'Total_Num', 'Bad_Num', 'Good_Num', 'Total_Pcnt', 'Bad_Rate', 'Good_Rate', 'Good_Pcnt',
         'Bad_Pcnt', 'Bad_Cum', 'Good_Cum', 'Woe', 'IV', 'Gini', 'KS']]
    df_new = df_new.reset_index(drop=True)
    return df_new
