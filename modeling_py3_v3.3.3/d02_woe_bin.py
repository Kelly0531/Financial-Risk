# -*- coding:utf-8 -*-
import logging
import logging.config
import pandas as pd
import d00_param
import imp

import tool
import numpy as np
import sys
from binning.drawwoe import draw_woe_fromdic
from binning.cumsum_cut import cut_cumsum_bin, chi_square_bin
from binning.best_ks_3 import Best_KS_Bin, apply_woe, filter_by_corr, func_tt_vardescrible

imp.reload(d00_param)
imp.reload(tool)


def _log_(log_file_name=None, stdout_on=False):
    log_file_name = log_file_name if log_file_name is not None else (__name__ + '.log')
    logger_ = logging.getLogger("riskengine")
    fmt = '[%(asctime)s.%(msecs)d][%(name)s][%(levelname)s]%(msg)s'
    date_fmt = '%Y-%m-%d %H:%M:%S'
    formatter = logging.Formatter(fmt=fmt, datefmt=date_fmt)
    if stdout_on:
        stout_handler = logging.StreamHandler(sys.stdout)
        stout_handler.setFormatter(formatter)
        logger_.addHandler(stout_handler)
    file_handler = logging.FileHandler(log_file_name, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger_.addHandler(file_handler)
    logger_.setLevel(logging.DEBUG)
    return logger_


def ex_inf_sum(list_x, method):
    list_y = [i for i in list_x if i not in [np.inf, -np.inf, 'inf', '-inf', np.nan, 'nan', 'null', 'NAN']]
    if method == 'SUM':
        return sum(list_y)
    if method == 'MAX':
        return max(list_y)


def cut_bin_choose(df_train, df_test, exclude_var, path_, tag, ex_1=False,
                   mono=True, manual=False, test_size=0.0, bin_ex_corr=False):
    '''
    :param df_train: 训练数据集[id,vars,tag]
    :param df_test: 测试数据集
    :param exclude_var: 剔除变量
    :param path_: 文件路径
    :param tag: Y标签
    :param ex_1: 是否对变量-1值剔除分箱
    :param enble_bin_way: 分箱方式：best_ks,uncut
    :param mono: 是否单调
    :param manual: 是否人工分箱
    :param bin_num: 分箱最大数
    :return:
    '''
    logger = _log_(log_file_name=path_ + "_execute.log", stdout_on=False)
    data = df_train.copy()
    all_vars = data.columns.tolist()

    data_var_type = pd.read_excel('./data/data_source.xlsx')
    Xpath_ = tool.prepare_path(project_name, 'd01_')
    df_describe = pd.read_excel(Xpath_ + '_x_describe_info.xlsx', sheet_name=u'特征探查分布情况')
    df_describe = df_describe[[u'特征中文名', u'覆盖率']]
    df_describe = df_describe.set_index(u'特征中文名').to_dict('index')
    data_dic = data_var_type.set_index('var_name').to_dict('index')
    str_vars = data_var_type[data_var_type['var_type'] == 'str']['var_name'].tolist()
    vars = list(set(all_vars) - set(exclude_var))
    dic_woe = {}
    dic_df = {}
    cant_col = []

    import pickle
    writer = pd.ExcelWriter(path_ + '_info.xlsx')
    ws1 = writer.book.add_worksheet('特征IV汇总')
    ws2 = writer.book.add_worksheet('特征分箱明细')
    title_Style = writer.book.add_format(tool.title_Style)
    cell_Style = writer.book.add_format(tool.cell_Style)
    num = 0
    summary = []
    # vars = ['devicetype','certi_city_level']
    var_dict = dict()
    cnt = 1
    print(10 * '*' + 'start auto woe bin.' + 10 * '*')
    for var_names in vars:
        logger.info('VAR_BIN PREPARE :%s' % var_names)
        print('VAR_BIN PREPARE :' + var_names)
        #
        if ex_1:
            data_cut = data[data[var_names] != -1]  # todo
        else:
            data_cut = data.copy()
        var_type = data_dic[var_names]['var_type']
        cut_bin_way = data_dic[var_names]['cut_bin_way']
        cut_bin_num = data_dic[var_names]['cut_bin_num']
        if cut_bin_way == 'best_ks':
            # flag_name Y标签 ; factor_name变量名 ; rate 分组占比不得小于的比例
            df_1 = Best_KS_Bin(data=data_cut[[var_names, tag]], flag_name=tag, factor_name=var_names,
                               not_in_list=['NaN', 'NAN'], rate=0.02, bin_num=cut_bin_num, var_type=var_type, mono=mono)

        elif cut_bin_way == 'uncut':
            df_1 = cut_cumsum_bin(data=data_cut[[var_names, tag]], flag_name=tag,
                                  factor_name=var_names, not_in_list=['NaN', 'NAN'],
                                  method='uncut', var_type=var_type,
                                  mono=False)
            print(df_1)
        elif var_type == 'number' and cut_bin_way in ['qcut', 'equidist_cut']:
            if cut_bin_way == 'qcut':
                df_1 = cut_cumsum_bin(data=data_cut[[var_names, tag]], flag_name=tag,
                                      factor_name=var_names, not_in_list=['NaN', 'NAN'],
                                      method='qcut', bin_num=cut_bin_num,
                                      mono=False)
            elif cut_bin_way == 'equidist_cut':
                df_1 = cut_cumsum_bin(data=data_cut[[var_names, tag]], flag_name=tag,
                                      factor_name=var_names, not_in_list=['NaN', 'NAN'],
                                      method='cut',
                                      mono=False, bin_num=cut_bin_num)
            else:
                print("cut_error")
        elif cut_bin_way == 'chi_square':
            df_1 = chi_square_bin(data=data_cut[[var_names, tag]], flag_name=tag,
                                  factor_name=var_names, not_in_list=['NaN', 'NAN'],
                                  mono=mono, bin_num=cut_bin_num, var_type=var_type)
        if len(df_1) == 0:
            print("NULL VAR")
        else:
            dic_df[var_names] = df_1
            df_1['var_name'] = [str(var_names) for _ in range(len(df_1))]
            woe_value = df_1['Woe']
            bin_name = df_1[var_names]
            dic_woe[var_names] = dict(zip(list(bin_name), list(woe_value)))
            summary.append([var_names, df_1.shape[0], ex_inf_sum(df_1['IV'], 'SUM'),
                            ex_inf_sum(df_1['KS'], 'MAX')])
            # df_1.to_excel(writer, 'detail', startrow=num)
            df_1 = df_1[[var_names, 'Total_Num', 'Bad_Num', 'Good_Num', 'Total_Pcnt', 'Bad_Rate', 'Woe',
                         'IV', 'Gini', 'KS', 'var_name']]
            df_1.columns = [var_names, '样本总数', '坏样本数', '好样本数', '分箱占比', '坏样本占比', 'Woe',
                            'IV', 'Gini', 'KS', 'var_name']
            rows = df_1.shape[0]
            df_1 = df_1.fillna('')
            ws2.write_row(num, 0, df_1.columns, title_Style)
            for ii in range(rows):
                ws2.write_row(1 + ii + num, 0, df_1.loc[ii,], cell_Style)
            if cnt == 1:
                ws2.set_column(0, 0, 20)
                ws2.set_column(1, df_1.shape[1], 10)
                cnt += 1
            var_dict[var_names] = num
            num += len(df_1) + 3
    f = open(path_ + '_vars_woe_bin_dic.pkl', 'wb+')
    pickle.dump(dic_woe, f)
    f.close()
    summary_df = pd.DataFrame(summary,
                              columns=['var', 'group_num', 'iv', 'ks'])
    summary_df = summary_df.sort_values('iv', ascending=False).reset_index(drop=True)
    summary_df['chn_meaning'] = summary_df['var'].apply(lambda x: data_dic[x]['chn_meaning'])
    summary_df['覆盖率'] = summary_df['var'].apply(lambda x: df_describe[x][u'覆盖率'])
    summary_df.columns = [u'特征简称', u'特征分箱数', 'IV', 'KS', u'特征中文名', u'覆盖率']
    fin_vars = list(summary_df[u'特征简称'])
    fin_vars.extend([key, tag])
    print(10 * '*' + 'finish auto woe bin.' + 10 * '*')
    if bin_ex_corr:
        print(10 * '*' + 'start to apply woe.' + 10 * '*')
        print(10 * '*' + 'wait....' + 10 * '*')
        # ##test
        # vars=[key, tag,'usermobileonlinedd','sex','credit_channel_name','register_channel_name']
        # ##

        m_data_train = apply_woe(df_train[fin_vars], dic_woe, str_vars, key, tag, path=path_)
        m_data_test = apply_woe(df_test[fin_vars], dic_woe, str_vars, key, tag, path=path_)
        print(10 * '*' + 'finish apply woe.' + 10 * '*')
        print(10 * '*' + 'start to exclude corr.' + 10 * '*')
        corr_limit = param_dic['corr_limit']
        nd_vars_woe = filter_by_corr(m_data_train, fin_vars, nec_vars=[], corr_limit=corr_limit)
        for i in nd_vars_woe:
            m_data_train[i] = m_data_train[i].astype(float)
            m_data_test[i] = m_data_test[i].astype(float)
        m_data_train_new = m_data_train[[ii for ii in m_data_train if ii[-4:] != '_bin']]
        m_data_test_new = m_data_test[[ii for ii in m_data_train if ii[-4:] != '_bin']]
        m_data_train.to_pickle('./data/' + project_name + '_m_data_train_bin.pkl')
        m_data_train_new.to_pickle('./data/' + project_name + '_m_data_train.pkl')
        m_data_test_new.to_pickle('./data/' + project_name + '_m_data_test.pkl')
        # nd_vars_bin = [i[:-4] + '_bin' for i in nd_vars_woe]
        # print(10 * '*' + 'start ttdescribe.' + 10 * '*')
        # if test_size > 0:
        #     func_tt_vardescrible(m_data_train, m_data_test, nd_vars_bin, path_, tag, 'model')
        # print(10 * '*' + 'finish ttdescribe.' + 10 * '*')
        nd_vars = [i[:-4] for i in nd_vars_woe]
        summary_df[u'是否因超过共线性阀值剔除'] = summary_df['特征简称'].apply(lambda x: 0 if x in set(nd_vars) else 1)
        print(10 * '*' + 'finish exclude corr.' + 10 * '*')
    # summary_df.to_excel(writer, 'summary', startrow=0)
    row_num = summary_df.shape[0]
    summary_df = summary_df.fillna('')
    ws1.write_row(0, 0, summary_df.columns, title_Style)
    for ii in range(row_num):
        ws1.write_row(1 + ii, 0, summary_df.loc[ii,], cell_Style)
        var_name = fin_vars[ii]
        start_num = var_dict[var_name] + 1
        link = "internal:'特征分箱明细'!A%s" % (str(start_num))  # 写超链接
        ws1.write_url(ii + 1, 0, link, string=var_name)
    ws1.set_column(0, 0, 25)
    ws1.set_column(1, summary_df.shape[1], 15)
    # df_train.describe().T.to_excel(writer, 'describe', startrow=0)
    writer.save()
    logger.info('VAR_BIN PREPARED , %s VARS CANNOT BE CUTTED' % str(len(cant_col)))
    logger.info('VAR_BIN DICT HAS BEEN PREPARED : %s' % path_ + '/' + project_name + '_vars_woe_bin_dic.pkl')

    # draw_woe_fromdic(dic_df, path_)
    # print 'woe字典已保存'
    # return dic_woe, summary_df


def cut_bin_choose_uncut(df_train, df_test, exclude_var, path_, tag, ex_1=False,
                         mono=True, manual=False, test_size=0.0, bin_ex_corr=False):
    '''
    :param df_train: 训练数据集[id,vars,tag]
    :param df_test: 测试数据集
    :param exclude_var: 剔除变量
    :param path_: 文件路径
    :param tag: Y标签
    :param ex_1: 是否对变量-1值剔除分箱
    :param enble_bin_way: 分箱方式：best_ks,uncut
    :param mono: 是否单调
    :param manual: 是否人工分箱
    :param bin_num: 分箱最大数
    :return:
    '''
    logger = _log_(log_file_name=path_ + "_execute.log", stdout_on=False)
    data = df_train.copy()
    all_vars = data.columns.tolist()
    Xpath_ = tool.prepare_path(project_name, 'd01_')
    df_describe = pd.read_excel(Xpath_ + '_x_describe_info.xlsx', sheet_name=u'特征探查分布情况')
    df_describe = df_describe[[u'特征中文名', u'覆盖率']]
    df_describe = df_describe.set_index(u'特征中文名').to_dict('index')
    data_var_type = pd.read_excel('./data/data_source.xlsx')
    data_dic = data_var_type.set_index('var_name').to_dict('index')
    str_vars = data_var_type[data_var_type['var_type'] == 'str']['var_name'].tolist()
    vars = list(set(all_vars) - set(exclude_var))
    dic_woe = {}
    dic_df = {}
    import xlsxwriter
    writer = xlsxwriter.Workbook(path_ + '_info_uncut.xlsx', {'nan_inf_to_errors': True})
    ws1 = writer.add_worksheet('特征IV汇总')
    ws2 = writer.add_worksheet('特征分箱明细')
    title_Style = writer.add_format(tool.title_Style)
    cell_Style = writer.add_format(tool.cell_Style)
    num = 0
    summary = []
    # vars = ['m3还款提醒事件']
    var_dict = dict()
    print(10 * '*' + 'start uncut woe bin.' + 10 * '*')
    for var_names in vars:
        logger.info('VAR_BIN PREPARE :%s' % var_names)
        print('VAR_BIN PREPARE :' + var_names)
        #
        if ex_1:
            data_cut = data[data[var_names] != -1]  # todo
        else:
            data_cut = data.copy()
        var_type = data_dic[var_names]['var_type']
        df_1 = cut_cumsum_bin(data=data_cut[[var_names, tag]], flag_name=tag,
                              factor_name=var_names, not_in_list=['NaN', 'NAN'],
                              method='uncut', var_type=var_type,
                              mono=False)
        if len(df_1) == 0:
            print("NULL VAR")
        else:
            dic_df[var_names] = df_1
            df_1['var_name'] = [str(var_names) for _ in range(len(df_1))]
            woe_value = df_1['Woe']
            bin_name = df_1[var_names]
            dic_woe[var_names] = dict(zip(list(bin_name), list(woe_value)))
            summary.append([var_names, df_1.shape[0], ex_inf_sum(df_1['IV'], 'SUM'),
                            ex_inf_sum(df_1['KS'], 'MAX')])
            # df_1.to_excel(writer, 'detail', startrow=num)
            rows = df_1.shape[0]
            df_1 = df_1.fillna('')
            ws2.write_row(num, 0, df_1.columns)
            for ii in range(rows):
                ws2.write_row(1 + ii + num, 0, df_1.loc[ii,])
            var_dict[var_names] = num
            num += len(df_1) + 3
    summary_df = pd.DataFrame(summary,
                              columns=['var', 'group_num', 'iv', 'ks'])
    summary_df = summary_df.sort_values('iv', ascending=False).reset_index(drop=True)
    summary_df['chn_meaning'] = summary_df['var'].apply(lambda x: data_dic[x]['chn_meaning'])
    summary_df['覆盖率'] = summary_df['var'].apply(lambda x: df_describe[x][u'覆盖率'])
    summary_df.columns = [u'特征简称', u'特征分箱数', 'IV', 'KS', u'特征中文名', u'覆盖率']
    fin_vars = list(summary_df[u'特征简称'])
    row_num = summary_df.shape[0]
    summary_df = summary_df.fillna('')
    ws1.write_row(0, 0, summary_df.columns, title_Style)
    for ii in range(row_num):
        ws1.write_row(1 + ii, 0, summary_df.loc[ii,], cell_Style)
        var_name = fin_vars[ii]
        start_num = var_dict[var_name] + 1
        link = "internal:'特征分箱明细'!A%s" % (str(start_num))  # 写超链接
        ws1.write_url(ii + 1, 0, link, string=var_name)
    ws1.set_column(0, 0, 25)
    ws1.set_column(1, summary_df.shape[1], 15)
    writer.close()
    print(10 * '*' + 'finish uncut woe bin.' + 10 * '*')


if __name__ == "__main__":
    with tool.Timer() as t:
        param_dic = d00_param.param_dic
        project_name = param_dic['project_name']
        path_ = tool.prepare_path(project_name, 'd02_')
        ex_cols = param_dic['ex_cols']
        tag = param_dic['tag']
        key = param_dic['key']
        test_size = param_dic['test_size']
        bin_ex_corr = param_dic['bin_ex_corr']
        mono = param_dic['mono']
        auto_bin_num = param_dic['auto_bin_num']
        nece_var = param_dic['nece_var']
        output_auto_cut_bin = param_dic['output_auto_cut_bin']
        output_uncut_bin = param_dic['output_uncut_bin']
        df_train = tool.read_file('./data/' + project_name + '_df_train.pkl')
        df_test = tool.read_file('./data/' + project_name + '_df_test.pkl')
        if output_auto_cut_bin:
            cut_bin_choose(df_train, df_test, ex_cols, path_, tag, mono=mono, test_size=test_size,
                           bin_ex_corr=bin_ex_corr)
        if output_uncut_bin:
            cut_bin_choose_uncut(df_train, df_test, ex_cols, path_, tag, mono=mono, test_size=test_size,
                                 bin_ex_corr=bin_ex_corr)
    print(t.secs)
