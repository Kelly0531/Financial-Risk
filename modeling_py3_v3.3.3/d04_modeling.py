# -*- coding:utf-8 -*-
import tool
import pandas as pd
import d00_param
import imp

imp.reload(d00_param)
import pickle
import numpy as np
from modeling.stepwise import func_sort_col, func_stepwise_1
from modeling.cal_ks_auc_score import cal_ks_test, func_report, gb_add_woe, func_tt_vardescrible
import time
from sklearn.externals import joblib
from binning.best_ks_3 import apply_woe, filter_by_corr
from tool import get_bin
import os


def model(model_ex_corr=True):
    start_time = time.time()
    param_dic = d00_param.param_dic
    project_name = param_dic['project_name']
    path_ = tool.prepare_path(project_name, 'd04_')
    tag = param_dic['tag']
    key = param_dic['key']
    corr_limit = param_dic['corr_limit']
    Xpath_ = tool.prepare_path(project_name, 'd02_')
    info = pd.read_excel(Xpath_ + '_info.xlsx', sheet_name='特征IV汇总')
    new_info = info[[u'特征简称', u'特征中文名', 'IV', u'覆盖率']]
    new_info_dict = new_info.set_index(u'特征简称').to_dict('index')
    vars = info.sort_values('IV', ascending=False)[:3000][u'特征简称'].tolist()
    fin_vars = vars
    # fin_vars = ['register_channel_name']
    fin_vars.extend([key, tag])
    if model_ex_corr and u'是否因超过共线性阀值剔除' in list(info.columns):
        m_data_train = tool.read_file('./data/' + project_name + '_m_data_train.pkl')
        m_data_test = tool.read_file('./data/' + project_name + '_m_data_test.pkl')
        nd_vars_woe = (info[info['是否因超过共线性阀值剔除'] == 0]['特征简称'] + '_woe').tolist()
    elif u'是否因超过共线性阀值剔除' in list(info.columns):
        m_data_train = tool.read_file('./data/' + project_name + '_m_data_train.pkl')
        m_data_test = tool.read_file('./data/' + project_name + '_m_data_test.pkl')
        nd_vars_woe = (info['特征简称'] + '_woe').tolist()
    elif model_ex_corr:
        f = open(Xpath_ + '_vars_woe_bin_dic.pkl', "rb")
        dic_woe = pickle.load(f)
        df_train = tool.read_file('./data/' + project_name + '_df_train.pkl')
        df_test = tool.read_file('./data/' + project_name + '_df_test.pkl')
        print(10 * '*' + 'start apply woe.' + 10 * '*')
        m_data_train = apply_woe(df_train[fin_vars], dic_woe, str_vars, key, tag, path=path_)
        m_data_test = apply_woe(df_test[fin_vars], dic_woe, str_vars, key, tag, path=path_)
        print(10 * '*' + 'finish apply woe.' + 10 * '*')
        nd_vars_woe = filter_by_corr(m_data_train, fin_vars, nec_vars=[], corr_limit=corr_limit)
        for i in nd_vars_woe:
            m_data_train[i] = m_data_train[i].astype(float)
            m_data_test[i] = m_data_test[i].astype(float)
        m_data_train.to_pickle('./data/' + project_name + '_m_data_train_bin.pkl')
        m_data_train = m_data_train[[ii for ii in m_data_train if ii[-4:] != '_bin']]
        m_data_test = m_data_test[[ii for ii in m_data_train if ii[-4:] != '_bin']]
        m_data_train.to_pickle('./data/' + project_name + '_m_data_train.pkl')
        m_data_test.to_pickle('./data/' + project_name + '_m_data_test.pkl')
    print(10 * '*' + 'finish vars check corr')
    sort_vars = func_sort_col(nd_vars_woe, m_data_train, tag)[1]
    train_cols, result, vars_dic, ks_df, error_var, odds, A, B = func_stepwise_1(sort_vars, m_data_train, tag,
                                                                                 pmml_btn=False, base_score=base_score,
                                                                                 double_score=double_score)
    # 保存模型
    joblib.dump(result, path_ + '_fin.model')
    # 加载模型
    # RF = joblib.load('rf.model')
    # df_gp_uncut, df_gp_cut, df_gp_qcut, ks_df, p, df_rs_score = cal_ks_test(result, m_data_test, tag)
    df_score = func_report(m_data_train, result, tag, base_score=base_score, double_score=double_score, save_path=path_,
                           test=m_data_test, info=None, odds=odds, score_detail_btn=False, data_dic=new_info_dict)
    root_path = path_.replace('d04_' + project_name, '')
    error_file = [ii for ii in os.listdir(root_path) if ii.find('_var_error_value.csv') >= 0]
    if error_file:
        for ii in error_file:
            df = tool.read_file(root_path + ii, header=-1)
            df.columns = ['var_name', 'value', 'flag']
            df.drop_duplicates(inplace=True)
            df.to_csv(root_path + ii, index=False)
        print(10 * '*' + '存在特征未匹配到分箱请查看文件%s' % root_path + ii + 10 * '*')
    df_score = df_score[[key, tag, 'score']]
    df_score['score_bin'] = df_score['score'].apply(lambda x: get_bin(x, score_right_list))
    df_score.to_csv(path_ + '_modeling_score.csv', index=False)
    print(10 * '*' + 'finish logstic modeling')
    print('spend times:', time.time() - start_time)


def model_tune(tune_del_vars=[], tune_nec_vars=[], is_all_necess=False, score_right_list=[]):
    start_time = time.time()
    param_dic = d00_param.param_dic
    project_name = param_dic['project_name']
    path_ = tool.prepare_path(project_name, 'd04_')
    tag = param_dic['tag']
    key = param_dic['key']
    corr_limit = param_dic['corr_limit']
    Xpath_ = tool.prepare_path(project_name, 'd02_')
    info = pd.read_excel(Xpath_ + '_info.xlsx', sheet_name='特征IV汇总')
    new_info = info[[u'特征简称', u'特征中文名', 'IV', u'覆盖率']]
    new_info_dict = new_info.set_index(u'特征简称').to_dict('index')
    vars = info.sort_values('IV', ascending=False)[:3000][u'特征简称'].tolist()
    m_data_train = tool.read_file('./data/' + project_name + '_m_data_train.pkl')
    m_data_test = tool.read_file('./data/' + project_name + '_m_data_test.pkl')
    if is_all_necess:
        fin_vars = tune_nec_vars
        nd_vars_woe = [i + "_woe" for i in fin_vars
                       if i + "_woe" in m_data_train.columns]
    else:
        fin_vars = filter(lambda x: x not in tune_del_vars, vars)
        if model_ex_corr:
            nd_vars_woe = filter_by_corr(m_data_train, fin_vars, nec_vars=tune_nec_vars, corr_limit=corr_limit)
        else:
            nd_vars_woe = [ii + "_woe" for ii in fin_vars if ii + "_woe" in m_data_train.columns]
    for i in nd_vars_woe:
        m_data_train[i] = m_data_train[i].astype(float)
        m_data_test[i] = m_data_test[i].astype(float)
    print(10 * '*' + 'finish vars check corr')
    sort_vars = func_sort_col(nd_vars_woe, m_data_train, tag)[1]
    train_cols, result, vars_dic, ks_df, error_var, odds, A, B = func_stepwise_1(sort_vars, m_data_train, tag,
                                                                                 pmml_btn=False, base_score=base_score,
                                                                                 double_score=double_score,
                                                                                 nec_vars=tune_nec_vars)
    # 保存模型
    joblib.dump(result, path_ + '_fin.model')
    df_score = func_report(m_data_train, result, tag, base_score=base_score, double_score=double_score, save_path=path_,
                           test=m_data_test, info=None, odds=odds, score_detail_btn=False, data_dic=new_info_dict)
    df_score = df_score[[key, tag, 'score']]
    df_score['score_bin'] = df_score['score'].apply(lambda x: get_bin(x, score_right_list))
    df_score.to_csv(path_ + '_modeling_score.csv', index=False)
    print(10 * '*' + 'odds:', odds)
    print(10 * '*' + 'finish logstic modeling')
    print('spend times:', time.time() - start_time)


def read_big_file(file_in):
    data = pd.read_csv(file_in, iterator=True)
    loop = True
    chunkSize = 100000
    chunks = []
    while loop:
        try:
            chunk = data.get_chunk(chunkSize)
            chunks.append(chunk)
        except StopIteration:
            loop = False
            print("Iteration is stopped.")
    df_train = pd.concat(chunks, ignore_index=True)
    return df_train


def output_score(file_in, model_vars, score_right_list=None):
    # encoding = tool.file_encoding(file_in)
    df = tool.read_file(file_in)
    param_dic = d00_param.param_dic
    project_name = param_dic['project_name']
    path_ = tool.prepare_path(project_name, 'd04_')
    tag = param_dic['tag']
    key = param_dic['key']
    fin_model_vars = model_vars
    fin_model_vars.extend([key, tag])
    Xpath_ = tool.prepare_path(project_name, 'd02_')
    f = open(Xpath_ + '_vars_woe_bin_dic.pkl', "rb")
    dic_woe = pickle.load(f)
    f = pd.read_excel(path_ + '_result.xlsx', sheet_name=u'评分卡刻度及公式')
    odds = float(f['odds'][0])
    print(10 * '*' + 'start to apply woe' + 10 * '*')
    m_data_cross = apply_woe(df[fin_model_vars], dic_woe, str_vars, key, tag, path=path_ + '_cross')
    print(10 * '*' + 'finish apply woe' + 10 * '*')
    m_data_train = tool.read_file('./data/' + project_name + '_m_data_train_bin.pkl')
    nd_vars_woe = [i + "_woe" for i in model_vars
                   if i + "_woe" in m_data_cross.columns]
    for i in nd_vars_woe:
        m_data_train[i] = m_data_train[i].astype(float)
        m_data_cross[i] = m_data_cross[i].astype(float)
    nd_vars_bin = [i[:-4] + '_bin' for i in nd_vars_woe]
    # 加载模型
    result = joblib.load(path_ + '_fin.model')
    df_gp_uncut, df_gp_cut, df_gp_qcut, ks_df, p, df_rs_score = cal_ks_test(result, m_data_cross, tag,
                                                                            base_score=base_score,
                                                                            double_score=double_score, odds=odds)
    writer = pd.ExcelWriter(path_ + '_cross_result.xlsx')
    title_Style = writer.book.add_format(tool.title_Style)
    cell_Style = writer.book.add_format(tool.cell_Style)
    # 跨时间集汇总指标
    ws_ks_df = writer.book.add_worksheet('跨时间集汇总指标')
    df_rs_score = df_rs_score[[key, tag, 'score']]
    df_rs_score['score_bin'] = df_rs_score['score'].apply(lambda x: get_bin(x, score_right_list))
    df_score = pd.read_csv(path_ + '_modeling_score.csv')
    model_ks, model_df = gb_add_woe(df_score, tag, 'artificial')
    cross_ks, cross_df = gb_add_woe(df_rs_score, tag, 'artificial')
    model_df = model_df[['bin', 'good', 'bad', 'bad_rate', 'bin_pct']]
    cross_df = cross_df[['bin', 'good', 'bad', 'bad_rate', 'bin_pct']]
    model_df.index = model_df['bin']
    cross_df.index = cross_df['bin']
    df_score_res = pd.concat([model_df, cross_df], axis=1, keys=['TRAIN', 'CROSS'], sort=False)
    df_score_res['PSI'] = (df_score_res[('TRAIN', 'bin_pct')] - df_score_res[('CROSS', 'bin_pct')]) * np.log(
        df_score_res[('TRAIN', 'bin_pct')] / df_score_res[('CROSS', 'bin_pct')])
    psi = sum([ii for ii in df_score_res['PSI'] if not pd.isnull(ii)])
    psi_df = pd.DataFrame({'describe': [psi]})
    psi_df.index = ['PSI']
    all_result = pd.concat([ks_df, psi_df])
    all_result = all_result.reset_index()
    all_result.columns = ['跨时间集', 'describe']
    print(all_result)
    df_score_res = df_score_res.reset_index(drop=True)
    base_bin = list(set(model_df['bin']) | set(cross_df['bin']))
    newbase = pd.DataFrame({'base_bin': base_bin})
    newbase = newbase.sort_values('base_bin')
    model_df = model_df.reset_index(drop=True)
    model_df_new = pd.merge(newbase, model_df, left_on='base_bin', right_on='bin', how='left')
    model_df_new.drop(columns=['bin'], inplace=True)
    cross_df = cross_df.reset_index(drop=True)
    cross_df_new = pd.merge(newbase, cross_df, left_on='base_bin', right_on='bin', how='left')
    cross_df_new.drop(columns=['bin'], inplace=True)
    row_num = all_result.shape[0]
    summary_df = all_result.fillna('')
    ws_ks_df.write_row(0, 0, summary_df.columns, title_Style)
    for ii in range(row_num):
        ws_ks_df.write_row(1 + ii, 0, summary_df.loc[ii,], cell_Style)
    ws_ks_df.set_column(0, summary_df.shape[1], 20)
    # 各分数段分布
    df_score_res.to_excel(writer, sheet_name='各分数段训练_跨时间集分布对比', startcol=0)
    df_rs_score.to_excel(writer, sheet_name='跨时间集打分明细', index=False)
    # 入模特征psi汇总指标及分箱明细对比
    func_tt_vardescrible(m_data_train, m_data_cross, writer, nd_vars_bin, tag)
    # 跨时间样本匹配不成功字段列表明细（如无匹配不成功则不输出文件）
    root_path = path_.replace('d04_' + project_name, '')
    error_file = [ii for ii in os.listdir(root_path) if ii.find('_var_error_value.csv') >= 0]
    if error_file:
        for ii in error_file:
            df = tool.read_file(root_path + ii, header=-1)
            df.columns = ['var_name', 'value', 'flag']
            df.drop_duplicates(inplace=True)
            df.to_csv(root_path + ii, index=False)
        print(10 * '*' + '存在特征未匹配到分箱请查看文件%s' % root_path + ii + 10 * '*')
    writer.save()


def output_score_psi(file_in, model_vars, score_right_list=None):
    df = tool.read_file(file_in)
    param_dic = d00_param.param_dic
    project_name = param_dic['project_name']
    path_ = tool.prepare_path(project_name, 'd04_')
    tag = param_dic['tag']
    key = param_dic['key']
    fin_model_vars = model_vars
    fin_model_vars.extend([key, tag])
    Xpath_ = tool.prepare_path(project_name, 'd02_')
    f = open(Xpath_ + '_vars_woe_bin_dic.pkl', "rb")
    dic_woe = pickle.load(f)
    print(10 * '*' + 'start to apply woe' + 10 * '*')
    m_data_cross = apply_woe(df[fin_model_vars], dic_woe, str_vars, key, tag, path=path_ + '_cross')
    print(10 * '*' + 'finish apply woe' + 10 * '*')
    m_data_train = tool.read_file('./data/' + project_name + '_m_data_train_bin.pkl')
    nd_vars_woe = [i + "_woe" for i in model_vars
                   if i + "_woe" in m_data_cross.columns]
    for i in nd_vars_woe:
        m_data_train[i] = m_data_train[i].astype(float)
        m_data_cross[i] = m_data_cross[i].astype(float)
    nd_vars_bin = [i[:-4] + '_bin' for i in nd_vars_woe]
    writer = pd.ExcelWriter(path_ + '_cross_result_vars_psi.xlsx')
    func_tt_vardescrible(m_data_train, m_data_cross, writer, nd_vars_bin, tag)
    root_path = path_.replace('d04_' + project_name, '')
    error_file = [ii for ii in os.listdir(root_path) if ii.find('_var_error_value.csv') >= 0]
    if error_file:
        for ii in error_file:
            df = tool.read_file(root_path + ii, header=-1)
            df.columns = ['var_name', 'value', 'flag']
            df.drop_duplicates(inplace=True)
            df.to_csv(root_path + ii, index=False)
        print(10 * '*' + '存在特征未匹配到分箱请查看文件%s' % root_path + ii + 10 * '*')
    writer.save()


if __name__ == "__main__":
    data_var_type = pd.read_excel('./data/data_source.xlsx')
    str_vars = set(data_var_type[data_var_type['var_type'] == 'str']['var_name'].tolist())
    param_dic = d00_param.param_dic
    project_name = param_dic['project_name']
    base_score = param_dic['base_score']
    double_score = param_dic['double_score']
    ex_cols = param_dic['ex_cols']
    nece_var = param_dic['nece_var']
    step = param_dic['step']
    is_all_necess = param_dic['is_all_necess']
    bin_ex_corr = param_dic['bin_ex_corr']
    model_ex_corr = param_dic['model_ex_corr']
    model_vars_psi = param_dic['model_vars_psi']
    path_ = tool.prepare_path(project_name, 'd04_')
    # 设定分数右边界list,默认左开右闭
    score_right_list = []
    for i in range(0, 31):
        score_right_list.append(10 * i + 400)
    # 弹出输入窗
    # try:
    #     model_step = int(input("model step:"))
    #     if model_step > 3 or model_step < 1:
    #         print ('input error ,please input int number in[1,2,3]')
    #         os._exit(0)
    # except:
    #     print('input error ,please input int number in[1,2,3]')
    #     os._exit(0)
    # # 首次训练模型
    if step == 1:
        print(10 * "*" + 'run model step %d' % step + 10 * "*")
        model(model_ex_corr=model_ex_corr)
        print(10 * "*" + 'finish model step %d' % step + 10 * "*")
    # # 调试模型
    if step == 2:
        print(10 * "*" + 'run model step %d' % step + 10 * "*")
        model_tune(tune_del_vars=ex_cols, tune_nec_vars=nece_var, is_all_necess=is_all_necess,
                   score_right_list=score_right_list)
        print(10 * "*" + 'finish model step %d' % step + 10 * "*")
    # # 跨时间样本打分
    if step == 3:
        print(10 * "*" + 'run model step %d' % step + 10 * "*")
        # model_vars = pd.read_csv(path_ + '_fin_model_vars.csv')
        # model_vars = model_vars['fin_vars'].tolist()
        f = pd.read_excel(path_ + '_result.xlsx', sheet_name=u'最终入模特征', skiprows=1)
        nan_index = list(f['特征简称']).index((np.nan))
        model_vars_list = [ii for ii in list(f['特征简称'])[:nan_index] if ii != 'intercept']
        file_in = './data/' + param_dic['cross_sample']
        output_score(file_in, model_vars_list, score_right_list)
        print(10 * "*" + 'finish model step %d' % step + 10 * "*")
    # # 跨时间样本变量psi
    if step == 4:
        print(10 * "*" + 'run model step %d' % step + 10 * "*")
        # model_vars_psi=[]
        file_in = './data/' + param_dic['cross_sample']
        output_score_psi(file_in, model_vars_psi)
        print(10 * "*" + 'finish model step %d' % step + 10 * "*")
