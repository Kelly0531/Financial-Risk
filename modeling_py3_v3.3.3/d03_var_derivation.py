# -*- coding:utf-8 -*-
import tool
import pandas as pd
import numpy as np
import d00_param
import imp
import copy

imp.reload(d00_param)
import time

from collections import Counter


def combine(l, n):
    '''
    :param l:需组合所有变量list
    :param n:组合变量数,两两、三三...
    :return:所有新组合的变量list
    '''
    answers = []
    one = [0] * n

    def next_c(li=0, ni=0):
        if ni == n:
            answers.append(copy.copy(one))
            return
        for lj in range(li, len(l)):
            one[ni] = l[lj]
            next_c(lj + 1, ni + 1)

    next_c()
    return answers


def rate_vars(data_i, var_list):
    new_var_name = '_'.join(var_list + ['rate'])
    data_i[new_var_name] = data_i.apply(lambda x: x[var_list[0]] / x[var_list[1]] if x[var_list[1]] != 0 else np.nan,
                                        axis=1)
    return data_i


def minus_vars(data_i, var_list):
    new_var_name = '_'.join(var_list + ['minus'])
    data_i[new_var_name] = data_i[var_list[0]] - data_i[var_list[1]]
    return data_i


def vals_format(val_list, agg):
    if agg in ['rate', 'minus']:
        if any(pd.isnull(val_list)):
            return np.nan
        elif agg == 'rate' and len(val_list) == 2:
            if float(val_list[0]) == 0.0 and float(val_list[1]) == 0.0:
                return 0
            elif float(val_list[1]) == 0.0:
                return np.inf
            else:
                return val_list[0] / val_list[1]
        elif agg == 'minus' and len(val_list) == 2:
            return val_list[0] - val_list[1]
        else:
            print('Error derivated')
    non_null_val = list(filter(lambda x: not pd.isnull(x), val_list))
    if len(non_null_val) == 0:
        return np.nan
    if agg == 'std':
        return np.std(non_null_val, ddof=1)
    elif agg == 'mean':
        return np.mean(non_null_val)
    elif agg in ['sum', 'min', 'max']:
        return eval(agg)(non_null_val)
    elif agg == 'same':
        cou = Counter(val_list)
        first = cou.most_common(1)
        return first[0][1]
    else:
        print('error agg')


def normal_vars(data_i, agg=[], var_list=[]):
    if len(agg) == 0 or len(var_list) <= 1:
        return data_i
    for agg_ii in agg:
        new_var_name = '_'.join(var_list + [agg_ii])
        print(new_var_name)
        data_i[new_var_name] = data_i.apply(
            lambda x: vals_format([x[var_list[ii]] for ii in range(len(var_list))], agg_ii), axis=1)
        # data_i[new_var_name] = data_i.apply(
        #     lambda x: np.nan if all(pd.isnull([x[var_list[ii]] for ii in range(len(var_list))])) else eval(agg_ii)(
        #         [x[var_list[ii]] for ii in range(len(var_list))]), axis=1)
    return data_i


def create_new_chn_name(new_var_info, var_info_dict, var_list, agg=[], cut_bin_num=5):
    for agg_ii in agg:
        new_dict = dict()
        new_dict['var_name'] = '_'.join(var_list + [agg_ii])
        new_dict['chn_name'] = '_'.join([var_info_dict[ii][1] for ii in var_list] + [agg_key_dict[agg_ii]])
        new_dict['var_type'] = 'number'
        new_dict['is_derivated'] = 1
        new_dict['cut_bin_way'] = 'best_ks'
        new_dict['cut_bin_num'] = cut_bin_num
        new_var_info.append(new_dict)
    return new_var_info


def derivate_vars(data_n, new_data_var_type, agg_list=[], col_list=[], cut_bin_num=5):
    '''
    :param data: 输入数据集
    :param agg_list: 衍生统计方法，如'sum','min','max','mean','std'等
    :param col_list:
    :return:
    '''
    # data = pd.DataFrame({'col1': [1, 2, 5, np.nan], 'col2': [21, 2, 5, np.nan], 'col3': [0, 2, 5, np.nan],
    #                    'col4': ['a', 'b', 'c', 'd']})
    # col_list=['col1', 'col2', 'col3']
    # agg_list=['sum', 'mean']
    result = data_n.copy()
    result = normal_vars(result, agg=agg_list, var_list=col_list)
    new_data_var_type = create_new_chn_name(new_data_var_type, chn_dict, col_list, agg=agg_list,
                                            cut_bin_num=cut_bin_num)
    # new_data = new_data[col_list].T
    # agg_list_spec = [ii for ii in agg_list if ii in ['minus', 'rate']]
    # 正常指标计算
    # agg_list_normal = [ii for ii in agg_list if ii not in ['minus', 'rate']]
    # if len(agg_list_normal)>0:
    #     new_col_names = ['_'.join(col_list + [ii]) for ii in agg_list_normal]
    #     new_result = new_data.agg(agg_list_normal).T
    #     new_result.columns = new_col_names
    #     new_data_var_type = create_new_chn_name(new_data_var_type, chn_dict, col_list, agg=agg_list_normal)
    #     if 'sum' in agg_list_normal:
    #         sum_var_name = '_'.join(col_list + ['sum'])
    #         is_null = [all(ii) for ii in np.array(new_data.isna().T)]
    #         new_result['col_isnull'] = is_null
    #         new_result[sum_var_name] = new_result.apply(lambda x: x[sum_var_name] if not x['col_isnull'] else np.nan,
    #                                                     axis=1)
    #         new_result.drop(columns=['col_isnull'], inplace=True)
    #     result = pd.concat([data_n, new_result], axis=1)
    # 特殊指标计算
    # if 'rate' in agg_list_spec:
    #     if len(col_list) == 2:
    #         result = rate_vars(result, col_list)
    #         new_data_var_type = create_new_chn_name(new_data_var_type, chn_dict, col_list, agg=['rate'])
    #     else:
    #         print('ERROR:', col_list, 'rate')
    # if 'minus' in agg_list_spec:
    #     if len(col_list) == 2:
    #         result = minus_vars(result, col_list)
    #         new_data_var_type = create_new_chn_name(new_data_var_type, chn_dict, col_list, agg=['minus'])
    #     else:
    #         print('ERROR:', col_list, 'minus')

    return result, new_data_var_type


if __name__ == "__main__":
    start_time = time.time()
    agg_key_dict = {'sum': u'求和',
                    'rate': u'相除',
                    'minus': u'相减',
                    'max': u'最大值',
                    'min': u'最小值',
                    'std': u'标准差',
                    'mean': u'平均值'
                    }
    param_dic = d00_param.param_dic
    project_name = param_dic['project_name']
    tag = param_dic['tag']
    key = param_dic['key']
    cut_bin_num = param_dic['auto_bin_num']
    datafile_input = param_dic['file_input']
    derivate_var_list = param_dic['derivate_var_list']
    derivate_agg_list = param_dic['derivate_agg_list']
    derivate_var_n = param_dic['derivate_var_n']
    output_data_file = param_dic['output_data_file']
    output_datasource_file = param_dic['output_datasource_file']
    data_var_type = pd.read_excel('./data/data_source.xlsx')
    # 输入数据集
    data = tool.read_file('./data/' + datafile_input)
    chn_dict = data_var_type.set_index('var_name').T.to_dict('list')
    # 输入衍生特征list及需衍生计算指标
    # derivate_var_list = ['age',
    #                      'usermobileonlinedd',
    #                      'certi_city_level',
    #                      'phone_city_level', ]
    # derivate_agg_list = ['sum', 'min', 'max', 'mean', 'minus', 'rate', 'std']
    # # 举例两两组合衍生即n=2
    # derivate_var_n = 2
    # # 输出衍生变量数据集路径
    # output_data_file = './data/data_derivated.csv'
    # # 输出衍生变量中文名路径
    # output_datasource_file = './data/data_source_derivated.csv'
    all_list = combine(derivate_var_list, derivate_var_n)
    new_data_var_type = []
    fin_data = data.copy()
    for var_l_i in all_list:
        fin_data, new_data_var_type = derivate_vars(fin_data, new_data_var_type, agg_list=derivate_agg_list,
                                                    col_list=var_l_i,
                                                    cut_bin_num=cut_bin_num)
    new_data_chn = pd.DataFrame(new_data_var_type)
    new_data_chn = new_data_chn[['var_name', 'var_type', 'chn_name', 'is_derivated', 'cut_bin_way', 'cut_bin_num']]
    new_data = fin_data[[key, tag] + list(new_data_chn['var_name'])]
    print(10 * '*' + '输出衍生变量数据集' + output_data_file + 10 * '*')

    new_data.to_csv(output_data_file, index=False, encoding='gbk')
    print(10 * '*' + '输出衍生变量中文名' + output_datasource_file + 10 * '*')
    new_data_chn.to_csv(output_datasource_file, index=False, encoding='gbk')
    print(10 * '*' + 'finish d03 data derivated' + 10 * '*')
    print('spend times:', time.time() - start_time)
