# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
import re
import math
import time
import itertools
from itertools import combinations
from numpy import array
from math import sqrt
from multiprocessing import Pool
import tool


def get_max_ks(date_df, start, end, rate, bad_name, good_name):
    ks = ''
    if end == start:
        return ks
    bad = date_df.loc[start:end, bad_name]
    good = date_df.loc[start:end, good_name]
    bad_good_cum = list(abs(np.cumsum(bad / sum(bad)) - np.cumsum(good / sum(good))))
    if bad_good_cum:
        ks = start + bad_good_cum.index(max(bad_good_cum))

    return ks


def cut_while_fun(start, end, piece, date_df, rate, bad_name, good_name, counts):
    '''
    :param start: 起始位置0
    :param end: 终止位置
    :param piece: 起始切分组
    :param date_df: 数据集
    :param rate: 最小分组占比
    :param bad_name: Y标签1对应列名
    :param good_name: Y标签0对应列名
    :param counts: 默认从1计数
    :return:
    '''
    point_all = []
    if counts >= piece or len(point_all) >= pow(2, piece - 1):
        return []
    ks_point = get_max_ks(date_df, start, end, rate, bad_name, good_name)
    if ks_point:
        if ks_point != '':
            t_up = cut_while_fun(start, ks_point, piece, date_df, rate, bad_name, good_name, counts + 1)
        else:
            t_up = []
        t_down = cut_while_fun(ks_point + 1, end, piece, date_df, rate, bad_name, good_name, counts + 1)
    else:
        t_up = []
        t_down = []
    point_all = t_up + [ks_point] + t_down
    return point_all


def ks_auto(date_df, piece, rate, bad_name, good_name):
    t_list = list(set(cut_while_fun(0, len(date_df) - 1, piece - 1, date_df, rate, bad_name, good_name, 1)))
    # py2
    # ks_point_all = [0] + filter(lambda x: x != '', t_list) + [len(date_df) - 1]
    # py3
    ks_point_all = [0] + list(filter(lambda x: x != '', t_list)) + [len(date_df) - 1]
    return ks_point_all


def get_combine(t_list, date_df, piece):
    t1 = 0
    t2 = len(date_df) - 1
    list0 = t_list[1:len(t_list) - 1]
    combine = []
    if len(t_list) - 2 < piece:
        c = len(t_list) - 2
    else:
        c = piece - 1
    list1 = list(itertools.combinations(list0, c))
    if list1:
        combine = map(lambda x: sorted(x + (t1 - 1, t2)), list1)

    return combine


def cal_iv(date_df, items, bad_name, good_name, rate, total_all, mono=True):
    iv0 = 0
    total_rate = [sum(date_df.ix[x[0]:x[1], bad_name] + date_df.ix[x[0]:x[1], good_name]) * 1.0 / total_all for x in
                  items]
    if [k for k in total_rate if k < rate]:
        return 0
    bad0 = array(list(map(lambda x: sum(date_df.ix[x[0]:x[1], bad_name]), items)))
    good0 = array(list(map(lambda x: sum(date_df.ix[x[0]:x[1], good_name]), items)))
    bad_rate0 = bad0 * 1.0 / (bad0 + good0)
    if 0 in bad0 or 0 in good0:
        return 0
    good_per0 = good0 * 1.0 / sum(date_df[good_name])
    bad_per0 = bad0 * 1.0 / sum(date_df[bad_name])
    woe0 = list(map(lambda x: math.log(x, math.e), good_per0 / bad_per0))
    if mono:
        if sorted(woe0, reverse=False) == list(woe0) and sorted(bad_rate0, reverse=True) == list(bad_rate0):
            iv0 = sum(woe0 * (good_per0 - bad_per0))
        elif sorted(woe0, reverse=True) == list(woe0) and sorted(bad_rate0, reverse=False) == list(bad_rate0):
            iv0 = sum(woe0 * (good_per0 - bad_per0))
    else:
        iv0 = sum(woe0 * (good_per0 - bad_per0))
    return iv0


def choose_best_combine(date_df, combine, bad_name, good_name, rate, total_all, mono=True):
    z = [0] * len(combine)
    for i in range(len(combine)):
        item = combine[i]
        z[i] = list(zip(map(lambda x: x + 1, item[0:len(item) - 1]), item[1:]))
    iv_list = list(map(lambda x: cal_iv(date_df, x, bad_name, good_name, rate, total_all, mono=mono), z))
    iv_max = max(iv_list)
    if iv_max == 0:
        return ''
    index_max = iv_list.index(iv_max)
    combine_max = z[index_max]
    return combine_max


def verify_woe(x):
    if re.match('^\-?\\d*\.?\d+$', str(x)):
        return x
    else:
        return 0


def best_df(date_df, items, na_df, factor_name, bad_name, good_name, total_all, good_all, bad_all, var_val_type):
    df0 = pd.DataFrame()
    if items:
        if var_val_type == 'number':
            right_max = list(map(lambda x: str(date_df.ix[x[1], factor_name]), items))
            piece0_old = list(map(
                lambda x: '(' + str(date_df.ix[x[0], factor_name]) + ',' + str(date_df.ix[x[1], factor_name]) + ')',
                items))
            # print(piece0_old)
            piece0_new = tool.get_bin_cate(right_max)
            piece0 = piece0_new
            # print(piece0_new)
        else:
            piece0 = list(map(
                lambda x: '(' + ','.join(date_df.ix[x[0]:x[1], factor_name]) + ')',
                items))
        bad0 = list(map(lambda x: sum(date_df.ix[x[0]:x[1], bad_name]), items))
        good0 = list(map(lambda x: sum(date_df.ix[x[0]:x[1], good_name]), items))
        if len(na_df) > 0:
            piece0 = array(
                list(piece0) + list(map(lambda x: '(' + str(x) + ',' + str(x) + ')', list(na_df[factor_name]))))
            bad0 = array(list(bad0) + list(na_df[bad_name]))
            good0 = array(list(good0) + list(na_df[good_name]))
        else:
            piece0 = array(list(piece0))
            bad0 = array(list(bad0))
            good0 = array(list(good0))
        total0 = bad0 + good0
        total_per0 = total0 * 1.0 / total_all
        bad_rate0 = bad0 * 1.0 / total0
        good_rate0 = 1 - bad_rate0
        good_per0 = good0 * 1.0 / good_all
        bad_per0 = bad0 * 1.0 / bad_all
        df0 = pd.DataFrame(
            list(zip(piece0, total0, bad0, good0, total_per0, bad_rate0, good_rate0, good_per0, bad_per0)),
            columns=[factor_name, 'Total_Num', 'Bad_Num', 'Good_Num', 'Total_Pcnt', 'Bad_Rate', 'Good_Rate',
                     'Good_Pcnt', 'Bad_Pcnt'])
        df0 = df0.sort_values(by='Bad_Rate', ascending=False)
        df0.index = range(len(df0))
        bad_per0 = array(list(df0['Bad_Pcnt']))
        good_per0 = array(list(df0['Good_Pcnt']))
        bad_rate0 = array(list(df0['Bad_Rate']))
        good_rate0 = array(list(df0['Good_Rate']))
        bad_cum = np.cumsum(bad_per0)
        good_cum = np.cumsum(good_per0)
        woe0 = list(map(lambda x: math.log(x, math.e), good_per0 / bad_per0))
        print(woe0)
        if 'inf' in str(woe0):
            woe0 = list(map(lambda x: verify_woe(x), woe0))
        print(woe0)
        iv0 = woe0 * (good_per0 - bad_per0)
        gini = 1 - pow(good_rate0, 2) - pow(bad_rate0, 2)
        df0['Bad_Cum'] = bad_cum
        df0['Good_Cum'] = good_cum
        df0["Woe"] = woe0
        df0["IV"] = iv0
        df0['Gini'] = gini
        df0['KS'] = abs(df0['Good_Cum'] - df0['Bad_Cum'])
    return df0


def all_information(date_df, na_df, piece, rate, factor_name, bad_name, good_name, total_all, good_all, bad_all,
                    var_val_type, mono=True):
    '''
        :param date_df:非异常值数据
        :param na_df 空值数据
        :param piece:切割组数
        :param rate:最小分组比例
        :param factor_name:变量名
        :param bad_name:坏的列名
        :param good_name:好的列名
        :param total_all:总样本数
        :param good_all:好的总样本数
        :param bad_all:坏的总样本数
        :return:
    '''
    p_sort = range(piece + 1)
    p_sort = [i for i in p_sort]
    p_sort.sort(reverse=True)
    t_list = ks_auto(date_df, piece, rate, bad_name, good_name)
    if not t_list:
        df1 = pd.DataFrame()
        print('Warning: this data cannot get bins or the bins does not satisfy monotonicity')
        return df1
    df1 = pd.DataFrame()
    for c in p_sort[:piece - 1]:
        combine = list(get_combine(t_list, date_df, c))
        best_combine = choose_best_combine(date_df, combine, bad_name, good_name, rate, total_all, mono=mono)
        df1 = best_df(date_df, best_combine, na_df, factor_name, bad_name, good_name, total_all, good_all, bad_all,
                      var_val_type)
        if len(df1) != 0:
            gini = sum(df1['Gini'] * df1['Total_Num'] / sum(df1['Total_Num']))
            print('piece_count:', str(len(df1)))
            print('IV_All_Max:', str(sum(df1['IV'])))
            print('Best_KS:', str(max(df1['KS'])))
            print('Gini_index:', str(gini))
            print(df1)
            return df1
    if len(df1) == 0:
        print('Warning: this data cannot get bins or the bins does not satisfy monotonicity')
        return df1


def verify_factor(x,value_type=[]):
    if x in ['NA', 'NAN', '', ' ', 'MISSING', 'NONE', 'NULL']:
        return 'NAN'
    if value_type=='number':
        if re.match('^\-?\d*\.?\d+$', x):
            x = float(x)
    return x


def path_df(path, sep, factor_name):
    data = pd.read_csv(path, sep=sep)
    data[factor_name] = data[factor_name].astype(str).map(lambda x: x.upper())
    data[factor_name] = data[factor_name].apply(lambda x: re.sub(' ', 'MISSING', x))

    return data


def verify_df_multiple(date_df, factor_name, total_name, bad_name, good_name):
    """
    :param date_df: factor_name,....
    :return: factor_name,good_name,bad_name
    """
    date_df = date_df.fillna(0)
    cols = date_df.columns
    if total_name in cols:
        date_df = date_df[date_df[total_name] != 0]
        if bad_name in cols and good_name in cols:
            date_df_check = date_df[date_df[good_name] + date_df[bad_name] - date_df[total_name] != 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print('Error: total amounts is not equal to the sum of bad & good amounts')
                print(date_df_check)
                return date_df
        elif bad_name in cols:
            date_df_check = date_df[date_df[total_name] - date_df[bad_name] < 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print('Error: total amounts is smaller than bad amounts')
                print(date_df_check)
                return date_df
            date_df[good_name] = date_df[total_name] - date_df[bad_name]
        elif good_name in cols:
            date_df_check = date_df[date_df[total_name] - date_df[good_name] < 0]
            if len(date_df_check) > 0:
                date_df = pd.DataFrame()
                print('Error: total amounts is smaller than good amounts')
                print(date_df_check)
                return date_df
            date_df[bad_name] = date_df[total_name] - date_df[good_name]
        else:
            print('Error: lack of bad or good data')
            date_df = pd.DataFrame()
            return date_df
        del date_df[total_name]
    elif bad_name not in cols:
        print('Error: lack of bad data')
        date_df = pd.DataFrame()
        return date_df
    elif good_name not in cols:
        print('Error: lack of good data')
        date_df = pd.DataFrame()
        return date_df
    date_df[good_name] = date_df[good_name].astype(int)
    date_df[bad_name] = date_df[bad_name].astype(int)
    date_df = date_df[date_df[bad_name] + date_df[good_name] != 0]
    date_df[factor_name] = date_df[factor_name].map(verify_factor)
    date_df = date_df.sort_values(by=[factor_name], ascending=True)
    date_df[factor_name] = date_df[factor_name].astype(str)
    if len(date_df[factor_name]) != len(set(date_df[factor_name])):
        df_bad = date_df.groupby(factor_name)[bad_name].agg([(bad_name, 'sum')]).reset_index()
        df_good = date_df.groupby(factor_name)[good_name].agg([(good_name, 'sum')]).reset_index()
        good_dict = dict(zip(df_good[factor_name], df_good[good_name]))
        df_bad[good_name] = df_bad[factor_name].map(good_dict)
        df_bad.index = range(len(df_bad))
        date_df = df_bad

    return date_df


def verify_df_two(data_df, flag_name, factor_name, good_name, bad_name,value_type):
    """
    :param data_df, factor_name, flag_name
    :return: factor_name,good_name,bad_name
    预处理：将df_按照flag(0-1)groupby，并按照var_value排序
    """

    data_df = data_df[-pd.isnull(data_df[flag_name])]
    if len(data_df) == 0:
        print('Error: the data is wrong')
        return data_df
    check = data_df[data_df[flag_name] > 1]
    if len(check) != 0:
        print('Error: there exits the number bigger than one in the data')
        data_df = pd.DataFrame()
        return data_df
    if flag_name != '':
        try:
            data_df[flag_name] = data_df[flag_name].astype(int)
        except:
            print('Error: the data is wrong')
            data_df = pd.DataFrame()
            return data_df
    data_df = data_df[flag_name].groupby(
        [data_df[factor_name], data_df[flag_name]]).count().unstack().reset_index().fillna(0)
    data_df.columns = [factor_name, good_name, bad_name]
    data_df[factor_name] = data_df[factor_name].apply(lambda x:verify_factor(x,value_type))
    # data_df_1=data_df[data_df[factor_name].apply(lambda x: str(x).find('NAN')>=0 or str(x).find('E')>=0)]
    # data_df_2=data_df[~data_df[factor_name].apply(lambda x: str(x).find('NAN')>=0 or str(x).find('E')>=0)]
    # data_df_2 = data_df_2.sort_values(by=[factor_name], ascending=True)
    # data_df=pd.concat([data_df_1,data_df_2])
    data_df.index = range(len(data_df))
    data_df[factor_name] = data_df[factor_name].astype(str)
    return data_df


def universal_df(data, flag_name, factor_name, total_name, bad_name, good_name,value_type):
    if flag_name != '':
        data = data[[factor_name, flag_name]]
        data = verify_df_two(data, flag_name, factor_name, good_name, bad_name,value_type)
    else:
        data = verify_df_multiple(data, factor_name, total_name, bad_name, good_name,value_type)
    return data


def Best_KS_Bin(path='', data=pd.DataFrame(), sep=',', flag_name='', factor_name='name', total_name='total',
                bad_name='bad', good_name='good', bin_num=5, rate=0.05, not_in_list=[], value_type=True,
                var_type='number', mono=True):
    """
    :param flag_name:Y标签
    :param factor_name: 变量名
    :param total_name:
    :param bad_name: bad
    :param good_name: good
    :param bin_num:切割组数,默认5
    :param rate:分组占比不得小于
    :param not_in_list:['NaN', '-1.0', '', '-1']
    :param value_type: True is numerical; False is nominal
    """
    # none_list = ['NA', 'NAN', '', ' ', 'MISSING', 'NONE', 'NULL']
    if path != '':
        # 若直接调用分箱则输入数据路径
        data = path_df(path, sep, factor_name)
    elif len(data) == 0:
        print('Error: there is no data')
        return data
    data = data.copy()
    data[factor_name] = data[factor_name].apply(lambda x: str(x).upper().replace(',', '_'))
    data = universal_df(data, flag_name, factor_name, total_name, bad_name, good_name,value_type)
    if len(data) == 0:
        return data
    good_all = sum(data[good_name])
    bad_all = sum(data[bad_name])
    total_all = good_all + bad_all
    if not_in_list:
        # 空值分组
        not_name = [str(k).upper() for k in not_in_list]
        # for n0 in none_list:
        #     print (n0)
        #     if n0 in not_name:
        #         not_name += ['NAN']  # todo
        #         print(not_name)
        #         break
        na_df = data[data[factor_name].isin(not_name)]
        if (0 in na_df[good_name]) or (0 in na_df[bad_name]):
            not_value = list(
                set(list(na_df[na_df[good_name] == 0][factor_name]) + list(na_df[na_df[bad_name] == 0][factor_name])))
            na_df = na_df.drop(na_df[na_df[factor_name].isin(not_value)].index)
            na_df.index = range(len(na_df))
        not_list = list(set(na_df[factor_name]))
        date_df = data[-data[factor_name].isin(not_list)]
    else:
        na_df = pd.DataFrame()
        date_df = data
    if len(date_df) == 0:
        print('Error: the data is wrong.')
        data = pd.DataFrame()
        return data
    if value_type:
        date_df = date_df.copy()
        if var_type != 'str':
            date_df[factor_name] = date_df[factor_name].apply(lambda x:verify_factor(x,var_type))
        type_len = set([type(k) for k in list(date_df[factor_name])])
        if len(type_len) > 1:
            str_df = date_df[date_df[factor_name].map(lambda x: type(x) == str)]
            number_df = date_df[date_df[factor_name].map(lambda x: type(x) == float)]
            number_df = number_df.sort_values(by=factor_name)
            str_df = str_df.sort_values(by=factor_name)
            date_df = str_df.append(number_df)
        else:
            date_df = date_df.sort_values(by=factor_name)
    else:
        date_df['bad_rate'] = date_df[bad_name] * 1.0 / (date_df[good_name] + date_df[bad_name])
        date_df = date_df.sort_values(by=['bad_rate', factor_name], ascending=False)
    date_df[factor_name] = date_df[factor_name].astype(str)  # todo
    date_df.index = range(len(date_df))
    # date_df.to_csv('./test.csv')
    # date_df.to_csv('11.csv', index=False)
    # print(date_df)
    # ks分箱
    bin_df = all_information(date_df, na_df, bin_num, rate, factor_name, bad_name, good_name, total_all, good_all,
                             bad_all, var_type, mono=mono)
    return bin_df

# df_1=Best_KS_Bin(rule,flag_name='c.is_overdue',factor_name='callintotdsstsumj150d')
#                  ,not_in_list=['NaN','-1.0','-2.0'])
# df_1 = Best_KS_Bin(data=rule[['callintotdsstsumj150d', 'c.is_overdue']], flag_name='c.is_overdue',
#                    factor_name='callintotdsstsumj150d', not_in_list=['NaN', '-1.0', '', '-1'])



def find_another_x(x, bin_dict):
    cc = {}
    for i in bin_dict.keys():
        if i not in ['(-1.0,-1.0)', '(NAN,NAN)']:
            cc[float(i.split(',')[0][1:])] = i
    dd = []
    for i in cc.keys():
        if i < x:
            dd.append(i)
    if dd == []:
        nd_key = min(cc.keys())
    else:
        nd_key = max(dd)
    bin_i = cc[nd_key]
    return bin_i


def assign_woe(x, dics, var_name,str_vars,path):
    '''
    :param x: 变量值
    :param dics:输入所需的变量woe_dic
    :param var_name: 变量名
    :return: 此函数用于训练集匹配woe，输入变量值，返回该变量对应的woe值;注意：若不采用BEST_KS的时候，需将区间改为左开右闭；
    '''
    if var_name in str_vars:
        x = str(x).upper().replace(',', '_')
    dic = dics[var_name]
    for i in dic.keys():
        # if 'NAN' in str(i) and str(x) in ['-999', '-999.0', '', '-1.0', '-1', 'nan']:
        if 'NAN' in str(i) and str(x) in [ 'NAN','nan']:
            y = dic[i]
            bin = i
            return y, bin
        # if i in ['(-1,-1)', '(-1.0,-1.0)', '[-1,-1]'] and x == -1:
        #     y = dic[i]
        #     bin = i
        else:
            if var_name in str_vars:
                if x in set(i[1:-1].split(',')):
                    y = dic[i]
                    bin = i
            else:
                try:
                    bin_num_1 = float(str(i).split(',')[0][1:].replace('(',''))
                    bin_num_2 = float(str(i).split(',')[1][:-1].replace(']',''))
                # from interval import Interval
                # zoom_bin = Interval(bin_num_1, bin_num_2, lower_closed=False)
                # if float(x) in zoom_bin:
                #     y = str(dic[i])
                #     bin = i
                    if float(bin_num_1) < float(x) <= float(bin_num_2):
                        y = str(dic[i])
                        bin = i
                except:
                    y = np.mean(list(dic.values()))
                    bin = '(-999,-999)'
                    error_str = ','.join([var_name, str(x), 'not in range'])
                    f = open(path + "_var_error_value.csv", "a")
                    f.write(error_str + '\n')
                    f.close()
    try:
        return y, bin
    except:
            # print 'try to save it '
            if var_name not in str_vars:
                bin = find_another_x(x, dic)
                y = dic[bin]
            else:
                y = np.mean(list(dic.values()))
                bin = '(-999,-999)'
            error_str = ','.join([var_name, str(x), 'not in range'])
            f = open(path + "_var_error_value.csv", "a")
            f.write(error_str + '\n')
            f.close()
            return float(y), bin


def apply_woe(data, woe_dic, str_vars, key, tag, cols=None,path=''):
    '''
    :param data: 原始数据['key','tag']
    :param woe_dic:
    :param str_vars:字符型特征list
    :return:m_data：注意：此m_data种包含所有变量，如只需入参变量，需后期筛选
    '''
    if cols:
        vars = cols
    else:
        vars = data.columns.tolist()
    vars_ = vars[:]
    vars_.extend([key, tag])
    nd_data = data[[key, tag]]
    # data_1 = data.fillna(-1)
    data_1=data.copy()
    nd_datas = nd_data.copy()
    error_var=[]
    for i in vars:
            if i in woe_dic.keys():
                nd_datas[i + '_woe'] = data_1[i].map(lambda x: assign_woe(x, woe_dic, i,str_vars,path)[0])
                nd_datas[i + '_bin'] = data_1[i].map(lambda x: assign_woe(x, woe_dic, i,str_vars,path)[1])
                nd_datas[i] = data_1[i][:]
    # for i in vars:
    #     ###
    #     try:
    #         if i in woe_dic.keys():
    #             nd_datas[i + '_woe'] = data_1[i].map(lambda x: assign_woe(x, woe_dic, i,str_vars,path)[0])
    #             nd_datas[i + '_bin'] = data_1[i].map(lambda x: assign_woe(x, woe_dic, i,str_vars,path)[1])
    #             nd_datas[i] = data_1[i][:]
    #     except Exception:
    #         error_var.append(i)
            # print (i, 'error mapped')
        ###
        # try:
        #     if i in woe_dic.keys():
        #         nd_datas[i + '_woe'] = data_1[i].map(lambda x: assign_woe(x, woe_dic, i)[0])
        #         nd_datas[i + '_bin'] = data_1[i].map(lambda x: assign_woe(x, woe_dic, i)[1])
        #         nd_datas[i] = data_1[i][:]
        # except Exception:
        #     print (i, 'error mapped')
    # if error_var:
    #     pd.DataFrame(error_var).to_csv(path+'_apply_woe_error_vars.csv',index=False,encoding='gbk')
    return nd_datas


def filter_by_corr(df_, series_, nec_vars=[], corr_limit=0.8):
    """
    df_: 包含WOE的数据(train)
    series_: Info表的var列，需要按优先级提前排序，默认IV
    corr_limit: 筛选的相关性阈值
    """
    series_ = [i for i in series_ if i not in nec_vars]
    series_ = [i + "_woe" for i in series_
               if i + "_woe" in df_.columns]
    nec_vars = [i + "_woe" for i in nec_vars if i + "_woe" in df_.columns]
    drop_set, var_set = set(), set(series_)
    for i in series_:
        if i in var_set:
            var_set.remove(i)
        if i not in drop_set:
            drop_set |= {v for v in var_set if np.corrcoef(
                df_[i].values.astype(float), df_[v].values.astype(float))[0, 1] > corr_limit}
            var_set -= drop_set
    return list(set([i for i in series_ if i not in drop_set]) | set(nec_vars))


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


def cal_ks_tt(df_rs, bin, tag):
    df_rs.ix[:, 'good'] = df_rs[tag].map(lambda x: func_good(x))
    df_rs.ix[:, 'bad'] = df_rs[tag].map(lambda x: func_bad(x))
    df_gp = df_rs.groupby([bin])['good', 'bad'].sum()
    bad_sum = df_gp['bad'].sum()
    good_sum = df_gp['good'].sum()
    bad_cnt = df_gp['bad'].cumsum()
    good_cnt = df_gp['good'].cumsum()
    df_gp.ix[:, 'bad_cnt'] = bad_cnt
    df_gp.ix[:, 'good_cnt'] = good_cnt
    df_gp['pct_bin'] = (df_gp['good'] + df_gp['bad']) / (bad_sum + good_sum)
    df_gp['pct_default'] = df_gp['bad'] / (df_gp['bad'] + df_gp['good'])
    df_gp['bad_pct'] = df_gp['bad'] / np.array([bad_sum for _ in range(len(df_gp))])
    df_gp['good_pct'] = df_gp['good'] / np.array([good_sum for _ in range(len(df_gp))])
    df_gp['Woe'] = [0 if 'inf' in str(ii) else ii for ii in np.log(df_gp['good_pct'] / df_gp['bad_pct'])]
    df_gp['IV'] = (df_gp['good_pct'] - df_gp['bad_pct']) * df_gp['Woe']
    df_gp.ix[:, 'b_c_p'] = df_gp['bad_cnt'] / np.array([bad_sum for _ in range(len(df_gp))])
    df_gp.ix[:, 'g_c_p'] = df_gp['good_cnt'] / np.array([good_sum for _ in range(len(df_gp))])
    df_gp.ix[:, 'ks'] = df_gp['g_c_p'] - df_gp['b_c_p']
    ks_max = max(df_gp['ks'].map(lambda x: abs(x)))
    iv = sum(df_gp['IV'].tolist())
    df_gp_new=df_gp[['good', 'bad', 'pct_bin', 'pct_default', 'Woe', 'IV', 'ks']]
    df_gp_new.columns=['good', 'bad', 'pct_bin', 'bad_rate', 'Woe', 'IV', 'ks']
    return ks_max, iv, df_gp_new


def func_tt_vardescrible(train, test, train_cols, save_path, tag, file_tag: str):
    path_ = save_path
    from pandas import ExcelWriter
    writer = ExcelWriter(path_ + '_train_test_compare_%s.xlsx' % file_tag)
    train_i = train.copy()
    test_i = test.copy()
    varname_list = []
    varks_list = []
    ks_j_ = []
    ks_i_ = []
    iv_i_ = []
    iv_j_ = []
    check = []
    group = []
    psi_all=[]
    num = 0
    for i in train_cols:
        print('turn to ', i)
        if i in train.columns and i != 'intercept':
            ks_i, iv_i, df_gp1 = cal_ks_tt(train_i, i, tag)
            ks_j, iv_j, df_gp2 = cal_ks_tt(test_i, i, tag)
            varname_list.append(i[:-4])
            varks_list.append(abs(ks_i - ks_j))
            ks_j_.append(ks_j)
            ks_i_.append(ks_i)
            iv_i_.append(iv_i)
            iv_j_.append(iv_j)
            group.append(df_gp1.shape[0])
            df_gp1 = df_gp1.reset_index()
            df_gp2 = df_gp2.reset_index()
            df_gp1.index = df_gp1[i]
            df_gp2.index = df_gp2[i]
            df_describle = pd.concat([df_gp1, df_gp2], axis=1, keys=['TRAIN', 'CROSS'], sort=False)
            df_describle=df_describle.reset_index(drop=True)
            df_describle['PSI'] = (df_describle[('TRAIN', 'pct_bin')] - df_describle[('CROSS', 'pct_bin')]) * np.log(
                df_describle[('TRAIN', 'pct_bin')] / df_describle[('CROSS', 'pct_bin')])
            psi = sum([ii for ii in df_describle['PSI'] if not pd.isnull(ii)])
            psi_all.append(psi)
            df_describle = df_describle.reset_index(drop=True)
            # df_describle = df_describle.sort_values(('TRAIN', 'Woe'))

            # 我从来没有想过会出现test不单调的情况，但是它居然出现了；没办法，只能加个判定了
            # ————————————判定开始————————————
            test_woe = df_describle['TEST']['Woe'].tolist()
            if pd.Series(test_woe).is_monotonic_decreasing or pd.Series(test_woe).is_monotonic_increasing:
                check.append(0)
            else:
                check.append(1)
            # ————————————判定结束————————————
            df_describle.to_excel(writer,'var_details', startrow=num )
            num += len(df_describle) + 4
    test_ks = pd.DataFrame({'var': varname_list,
                            'ks_train': ks_i_,
                            'ks_test': ks_j_,
                            'ks_dif': varks_list,
                            'iv_train': iv_i_,
                            'iv_test': iv_j_,
                            'check': check,
                            'group': group,
                            'PSI': psi_all
                            })
    ks_sort = test_ks.sort_values('ks_test', ascending=False)[[
        'var', 'iv_train', 'iv_test', 'ks_train', 'ks_test', 'ks_dif', 'group', 'check','PSI']]
    ks_sort.to_excel(writer, 'summary', startrow=0)
    writer.save()
    # return ks_sort
