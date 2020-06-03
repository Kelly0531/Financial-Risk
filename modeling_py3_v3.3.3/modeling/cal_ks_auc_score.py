# -*- coding:utf-8 -*-

import pandas as pd
import statsmodels.api as sm
# import pylab as pl
import numpy as np
import math
# import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.linear_model import LogisticRegression
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from binning.drawwoe import draw_woe
from tool import corr_img
import tool
import imp
imp.reload(tool)

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


# def group_score(data,code):
#     df_rs=data.copy()
#     list=df_rs.index.tolist()
#     df_rs.ix[:,'no']=list
#     a=pd.qcut(df_rs['p'],10)
#     df_rs.ix[:,'p_group']=a
#     df_rs['good']=df_rs[code].map(lambda x:func_good(x))
#     df_rs['bad']=df_rs[code].map(lambda x:func_bad(x))
#     b=df_rs.groupby(['p_group',])['no','good','bad'].sum()
#     df_gp=pd.DataFrame(b)
#     return df_gp

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


# df_score_gp=group_score(final_score,'tag','reborrow_score')


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


# df_score['score_group']=df_score['score'].map(lambda x : catch_score(x,group))


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


# df_gp=gb_add_woe(data,code,'uncut',score_name=i)


def cal_score(data, base_score, double_score, odds):
    '''
    :param data: df_['id','tag'] 为了方便要统一一下字段名字_(:зゝ∠)_
    :param odds:  全局bad/good
    :param result:模型训练结果，以result格式
    :return:每个客户的总分
    '''
    B = double_score / math.log(2)
    A = base_score + B * math.log(odds)
    df_rs = data.copy()
    df_rs['p_'] = df_rs['p'].apply(lambda x: 1 - float(x))
    df_rs['score'] = np.array([A for _ in range(len(df_rs))]) + \
                     np.array([B for _ in range(len(df_rs))]) * np.log(df_rs.ix[:, 'p'] / df_rs.ix[:, 'p_'])
    # df_rs['score'].astype(int)
    return df_rs, A, B


def cal_scorei(woe_dic, base_score, double_score, odds, result):
    '''
    :param woe_dic: 变量的woe字典,varname
    :param base_score:
    :param double_score:
    :param odds: 全局good/bad
    :param result:模型训练结果，以result格式,varname_woe
    :return:返回每个入参变量的小分
    '''
    B = double_score / math.log(2)
    A = base_score + B * math.log(odds)
    Intercept = result.params['intercept']
    N = len(result.params) - 1
    summary = []
    # print('prepared')
    for i in result.params.index.tolist():
        if i != 'intercept':
            coef_i = result.params[i]
            dic_i = woe_dic[i[:-4]]
            for j in dic_i.keys():
                woe_i = dic_i[j]
                score_i = (woe_i * coef_i + Intercept / N) * B + A / N
                summary.append([i, j, score_i])
        summary.append(['', '', ''])
    summary_df = pd.DataFrame(summary, columns=['var', 'bin', 'score'])
    return summary_df


# cal_scorei(dic_all,550,40,30,result1)

#
#
# col_1,result,vars_dic=func_stepwise_1(var,mm,'tag')
#

# summary_df=cal_scorei(xhd_woe_dic,580,50,0.1,result)


def cal_bin_num(bin, ):
    pass


def cal_PSI_var(train, test, train_vars, save_path):
    '''
    :param data1:使用训练集的df_gp['bin','amount']
    :param data2:使用测试集的df_score(总分)
    :return: 计算两个data的PSI，
    ：psi = sum(（实际占比-预期占比）*ln(实际占比/预期占比))
    '''
    psi = {}
    from pandas import ExcelWriter
    writer = ExcelWriter(save_path + '_PSI_INFO.xlsx')
    num = 0
    for var_name in train_vars:
        train_i = train[var_name].value_counts(True)
        test_i = test[var_name].value_counts(True)
        a = pd.DataFrame(train_i)
        a['test_pct'] = test_i
        a.columns = ['train_pct', 'test_pct']
        a['var_name'] = var_name

        a['psi'] = (a['train_pct'] - a['test_pct']) * np.log(a['train_pct'] / a['test_pct'])
        a.to_excel(writer, 'detail', startrow=num)
        num += len(a) + 3
        psi[var_name] = sum(a['psi'])
    PSI_DF = pd.DataFrame(psi, index=['PSI']).T
    PSI_DF.to_excel(writer, 'psi_summary', startrow=0)
    writer.save()
    return PSI_DF


# def func_cal(data,key,tag,vars,base_score,double_score,woe_dic,save_path,method,enable_test=False):
#     '''
#     :param data: 客户数据['id','tag']
#     :param X: data[train_cols]
#     :param Y: data[tag]
#     :param base_score:
#     :param woe_dic:m_data所返回的变量woe字典
#     :param v: odds
#     :return: 根据已选出来的变量，生成模型结果报告
#     '''
#     from pandas import ExcelWriter
#     writer=ExcelWriter(save_path)
#     rs=data.copy()
#     Y=data[tag]
#     print 'stage1 : 逻辑回归结果：'
#     if method=='REPORT':
#         X=data[vars]
#         logit = sm.Logit(Y,X)
#         result1 = logit.fit()
#     elif method=='STEPWISE':
#         train_vars = data.columns.tolist()
#         train_vars.remove(key)
#         train_vars.remove(tag)
#         if 'Unnamed: 0' in train_vars:
#             train_vars.remove('Unnamed: 0')
#         # X = data[train_vars]
#         col_1, result1, vars_dic = func_stepwise_1(train_vars, data, tag)
#     print result1.summary()
#     p = result1.predict().tolist()
#     auc = roc_auc_score(Y, p)
#     rs['p']=p
#     odds=float(len(Y[Y==0]))/float(len(Y[Y==1]))
#     print 'stage2: 开始计算小分：'
#     # score_i_df=cal_scorei(woe_dic, base_score, double_score, odds, result1)
#     # score_i_df.to_excel(writer, 'score_i', startrow=0)
#     print 'stage3: 开始计算单个客户得分：'
#     df_rs_score = cal_score(rs, base_score, double_score, odds)
#     df_rs_score.to_excel(writer,'cust_score',startrow=0)
#     print df_rs_score[:3]
#     print 'stage4: 开始计算KS：'
#     ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score,tag,'uncut')
#     ks_max_cut, df_gp_cut = gb_add_woe(df_rs_score,tag,'cut')
#     ks_max_qcut, df_gp_qcut = gb_add_woe(df_rs_score,tag,'qcut')
#     ks=[auc,ks_max_uncut,ks_max_cut,ks_max_qcut]
#     df_gp_cut.to_excel(writer,'df_gp_score',startrow=0)
#     df_gp_qcut.to_excel(writer,'df_gp_score',startrow=len(df_gp_cut)+3)
#     ks_df=pd.DataFrame(ks,columns=['describe'],index=['auc','ks_max_uncut','ks_max_cut','ks_max_qcut'])
#     ks_df.to_excel(writer,'result_summary',startrow=0)
#     writer.save()
#     print '--------AUC: '+str(auc)+'---------'
#     print '---------------KS: -----------------'
#     print ks_df
#     print '----------------------------------'
#     print df_gp_cut
#     return  result1,rs,auc,df_gp_cut


def cal_ks(result1, data, tag, base_score, double_score):
    '''
    :param data: 客户数据['id','tag']
    :param X: data[train_cols]
    :param Y: data[tag]
    :param base_score:
    :param woe_dic:m_data所返回的变量woe字典
    :param v: odds
    :return: 根据已选出来的变量，生成模型结果报告
    '''
    rs = data.copy()
    Y = data[tag]
    print(result1.summary2())
    p = result1.predict().tolist()
    auc = roc_auc_score(Y, p)
    rs['p'] = p
    odds = float(len(Y[Y == 0])) / float(len(Y[Y == 1]))
    # print 'stage3: 开始计算单个客户得分：'
    df_rs_score, A, B = cal_score(rs, base_score, double_score, odds)
    # print 'stage4: 开始计算KS：'
    ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score, tag, 'uncut')
    ks_max_cut, df_gp_cut = gb_add_woe(df_rs_score, tag, 'cut')
    ks_max_qcut, df_gp_qcut = gb_add_woe(df_rs_score, tag, 'qcut')
    ks = [auc, ks_max_uncut, ks_max_cut, ks_max_qcut]
    ks_df = pd.DataFrame(ks, columns=['describe'], index=['auc', 'ks_max_uncut', 'ks_max_cut', 'ks_max_qcut'])
    print('--------AUC: ' + str(auc) + '---------')
    print('---------------KS: -----------------')
    print(ks_df)
    return ks_df, odds, A, B


def cal_ks_test(result, test_data, tag, base_score=550, double_score=-40, odds=30):
    '''
    :param data: 客户数据['id','tag']
    :param X: data[train_cols]
    :param Y: data[tag]
    :param base_score:
    :param woe_dic:m_data所返回的变量woe字典
    :param v: odds
    :return: 根据已选出来的变量，生成模型结果报告
    '''
    rs = test_data.copy()
    predict_cols = result.params.index.tolist()
    rs['intercept'] = 1.0
    p = result.predict(rs[predict_cols])
    rs['p'] = p
    Y = rs[tag]
    auc = roc_auc_score(Y, p)
    df_rs_score, A, B = cal_score(rs, base_score, double_score, odds)
    ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score, tag, 'uncut')
    ks_max_cut, df_gp_cut = gb_add_woe(df_rs_score, tag, 'cut')
    ks_max_qcut, df_gp_qcut = gb_add_woe(df_rs_score, tag, 'qcut')
    ks = [auc, ks_max_uncut, ks_max_cut, ks_max_qcut]
    ks_df = pd.DataFrame(ks, columns=['describe'], index=['AUC', 'KS_不切分', 'KS_等距切分', 'KS_等频切分'])
    # print('--------AUC: ' + str(auc) + '---------')
    # print('---------------KS: -----------------')
    # print(ks_df)
    # print('---------------cut:-----------------')
    # print (df_gp_cut)
    # return df_rs_score
    return df_gp_uncut, df_gp_cut, df_gp_qcut, ks_df, p, df_rs_score


def func_report(data, result, tag, base_score, double_score, save_path, test, info=None, odds=30,
                score_detail_btn=False, data_dic=dict()):
    '''
    :param data: 训练集
    :param result: 模型训练结果
    :param tag: Y标签
    :param base_score: 基准分
    :param double_score: 翻倍分数
    :param save_path: 存储路径，默认当前目录下，不带.xlsx
    :param test: 测试集
    :param info: info表的detail页签
    :param odds: 翻倍对应好坏比例，默认30:1
    :param score_detail_btn: 是否输出模型分明细
    :return:
    '''
    from pandas import ExcelWriter
    writer = pd.ExcelWriter(save_path + '_result.xlsx')

    title_Style = writer.book.add_format(tool.title_Style)
    cell_Style = writer.book.add_format(tool.cell_Style)
    # formats = '#,##0.00'
    rs = data.copy()
    ts_data=test.copy()
    Y_train = data[tag]
    Y_test= test[tag]
    ts_data['intercept']=1.0
    print('stage1 : LOGISTIC RESULT:')
    print(result.summary2())
    train_col = result.params.index.tolist()
    p_train = result.predict(data[train_col]).tolist()
    p_test = result.predict(ts_data[train_col]).tolist()
    auc_train = roc_auc_score(Y_train, p_train)
    # auc_test = roc_auc_score(Y_test, p_test)
    # roc
    from sklearn import metrics
    from sklearn.metrics import roc_curve
    from matplotlib import pyplot as plt
    from pylab import mpl
    mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
    mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
    lr_fpr_train, lr_tpr_train, lr_thresholds = roc_curve(Y_train, p_train)
    lr_fpr_test, lr_tpr_test, lr_thresholds = roc_curve(Y_test, p_test)
    # 分别计算这两个模型的auc的值, auc值就是roc曲线下的面积
    lr_roc_auc_train = metrics.auc(lr_fpr_train, lr_tpr_train)
    lr_roc_auc_test = metrics.auc(lr_fpr_test, lr_tpr_test)
    plt.figure(figsize=(8, 8))
    plt.plot([0, 1], [0, 1], '--', color='r')
    plt.plot(lr_fpr_train, lr_tpr_train, label='训练集(AUC = %0.2f)' % lr_roc_auc_train)
    plt.plot(lr_fpr_test, lr_tpr_test, label='测试集(AUC = %0.2f)' % lr_roc_auc_test)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title('ROC')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.legend()
    ROC_IMG=save_path + "_ROC.png"
    plt.savefig(ROC_IMG)
    #
    rs['p'] = p_train
    # odds=float(len(Y[Y==0]))/float(len(Y[Y==1]))
    # odds=30
    print('stage1.5:PREPARE VAR RESULT：')
    num = 0
    # tt = func_tt_vardescrible(data, test, [i[:-4] + '_bin' for i in train_col if '_woe' in i], save_path[:-5], tag)
    print('stage3: BEGIN TO CAL SCORE_I:()')
    df_rs_score1, A, B = cal_score(rs, base_score, double_score, odds)
    # df_rs_score1.to_excel(writer,sheetname='score_detail')
    print('stage4: BEGIN TO CAL KS ：')
    ks_max_uncut, df_gp_uncut = gb_add_woe(df_rs_score1, tag, 'uncut')
    ks_max_cut, df_gp_cut = gb_add_woe(df_rs_score1, tag, 'cut')
    ks_max_qcut, df_gp_qcut = gb_add_woe(df_rs_score1, tag, 'qcut')
    ks = [auc_train, ks_max_uncut, ks_max_cut, ks_max_qcut]

    ks_df = pd.DataFrame(ks, columns=['describe'], index=['AUC', 'KS_不切分', 'KS_等距切分', 'KS_等频切分'])
    df_gp_uncut2, df_gp_cut2, df_gp_qcut2, ks_df2, p2, df_rs_score2 = cal_ks_test(result, test, tag, base_score,
                                                                                  double_score, odds)
    ks_df=ks_df.reset_index()
    ks_df.columns=['训练集','describe']
    ks_df2 = ks_df2.reset_index()
    ks_df2.columns = ['测试集', 'describe']
    # --------------------------------
    df_score = df_rs_score1.append(df_rs_score2)
    # 模型AUC/KS结果
    # ks_df.to_excel(writer, '模型AUC_KS结果', startrow=0, index=False)
    ws_ks_df = writer.book.add_worksheet('模型AUC_KS结果')
    ws1 = writer.book.add_worksheet('最终入模特征')
    ws2 = writer.book.add_worksheet('评分卡刻度及公式')
    ws3 = writer.book.add_worksheet('模型各特征分箱明细（自动分箱）')
    row_num = ks_df.shape[0]
    summary_df = ks_df.fillna('')
    ws_ks_df.write_row(0, 0, summary_df.columns, title_Style)
    for ii in range(row_num):
        ws_ks_df.write_row(1 + ii, 0, summary_df.loc[ii,], cell_Style)
    row_num = ks_df2.shape[0]
    summary_df = ks_df2.fillna('')
    ws_ks_df.write_row(3+ks_df.shape[0], 0, summary_df.columns, title_Style)
    for ii in range(row_num):
        ws_ks_df.write_row(4+ks_df.shape[0] + ii, 0, summary_df.loc[ii,], cell_Style)
    ws_ks_df.set_column(0, summary_df.shape[1], 20)
    ws_ks_df.insert_image(6+ks_df.shape[0] + row_num, 0, ROC_IMG)
    # ks_df2.to_excel(writer, '模型AUC_KS结果', startrow=ks_df.shape[0] + 3, index=False)
    ks, gp = gb_add_woe(df_score, tag, '10score_cut')
    df_title = pd.DataFrame(columns=[u'模型分等距分箱'])
    df_title.to_excel(writer, '训练集各分数段分布情况', startrow=0)
    df_gp_cut.to_excel(writer, '训练集各分数段分布情况', startrow=1)
    df_title = pd.DataFrame(columns=[u'模型分等频分箱'])
    df_title.to_excel(writer, '训练集各分数段分布情况', startrow=df_gp_cut.shape[0] + 3)
    df_gp_qcut.to_excel(writer, '训练集各分数段分布情况', startrow=df_gp_cut.shape[0] + 4)
    df_title = pd.DataFrame(columns=[u'模型分等距分箱'])
    df_title.to_excel(writer, '测试集各分数段分布情况', startrow=0)
    df_gp_cut2.to_excel(writer, '测试集各分数段分布情况', startrow=1)
    df_title = pd.DataFrame(columns=[u'模型分等频分箱'])
    df_title.to_excel(writer, '测试集各分数段分布情况', startrow=df_gp_cut.shape[0] + 3)
    df_gp_qcut2.to_excel(writer, '测试集各分数段分布情况', startrow=df_gp_cut.shape[0] + 4)
    gp.to_excel(writer, '训练&测试集汇总10分档分布情况', startrow=0)

    result_df = pd.DataFrame({'params': list(result.params.index),
                              'coef': list(result.params),
                              'p_value': list(result.pvalues)})
    result_df['特征简称'] = result_df['params'].apply(lambda x: x[:-4] if x != 'intercept' else x)
    result_df['特征中文名'] = result_df['特征简称'].apply(
        lambda x: data_dic[x][u'特征中文名'] if x != 'intercept' else np.nan)
    result_df['IV'] = result_df['特征简称'].apply(
        lambda x: data_dic[x]['IV'] if x != 'intercept' else np.nan)
    result_df[u'覆盖率'] = result_df['特征简称'].apply(
        lambda x: data_dic[x][u'覆盖率'] if x != 'intercept' else np.nan)
    result_df.sort_values('IV', ascending=False, inplace=True)
    # result_df.to_excel(writer, 'Final Model Vars')
    # 输出评分明细
    if score_detail_btn:
        df_score.to_excel(writer, 'modeling_score_details')
    xpath_ = save_path.replace( 'd04_','d02_')
    info_dt = pd.read_excel(xpath_ + '_info.xlsx', sheet_name='特征分箱明细')
    if info:
        var_list = [i[:-4] for i in result.params.index if '_woe' in i]
        draw_woe(info, var_list, save_path)
    # 评分卡公式参数
    df_param = pd.DataFrame([{'odds': odds, 'A': A, 'B': B}])
    row_num = df_param.shape[0]
    summary_df = df_param.fillna('')
    ws2.write_row(0, 0, summary_df.columns,title_Style)
    for ii in range(row_num):
        ws2.write_row(1 + ii, 0, summary_df.loc[ii,],cell_Style)
    ws2.set_column(0, df_param.shape[1], 20)
    base_str1=''
    for ii in result_df['特征简称']:
        if ii != 'intercept':
            base_str1=base_str1+"+ %s * %s "%(ii+'_coef',ii+'_woe')+'\n'
        else:
            base_str1=base_str1+"+ intercept "+'\n'
    fin_str1="Score = A + B * [ %s ]"%(base_str1)
    ws2.write(row_num + 2, 0, '评分卡公式')
    ws2.write(row_num + 3, 0, fin_str1)
    # f = open(save_path + '_fin_model_parms.txt', 'w')
    # f.write(','.join(['odds', 'A', 'B']) + '\n')
    # f.write(','.join([str(odds), str(A), str(B)]))
    # f.close()
    # out_vars = [u'm12验证码通知平台']
    var_dict=dict()
    # 最终入模特征相关性热力图
    out_vars = [ii for ii in result_df[u'特征简称'] if ii != 'intercept']
    out_img = []
    if len(out_vars) >= 1:
        df_corr = data[[ii + '_woe' for ii in out_vars]]
        df_corr.columns=out_vars
        dfData = df_corr.corr()
        dfData = dfData.reset_index()
        out_img = corr_img(df_corr, save_path)
    # 最终入模特征分箱明细
    coef_dic = result_df[[u'特征简称','coef']].set_index(u'特征简称').to_dict('index')
    startrow=0
    for var_i in out_vars:
        info_dt_i = info_dt[info_dt[u'var_name'] == var_i]
        info_dt_i.columns = [var_i] + list(info_dt_i.columns[1:])
        coef=coef_dic[var_i]['coef']
        info_dt_i=info_dt_i.copy()
        info_dt_i=info_dt_i.reset_index(drop=True)
        info_dt_i[u'分箱对应打分']=coef*info_dt_i['Woe']*B
        var_dict[var_i] = startrow
        row_num = info_dt_i.shape[0]
        summary_df = info_dt_i.fillna('')
        ws3.write_row(startrow, 0, summary_df.columns, title_Style)
        for ii in range(row_num):
            ws3.write_row(1 + ii+startrow, 0, summary_df.loc[ii,], cell_Style)
        # info_dt_i.to_excel(writer, 'Final Model Vars Details(auto)', startrow=startrow,index=False)
        startrow=startrow+info_dt_i.shape[0]+3
    ws3.set_column(0, 0, 25)
    ws3.set_column(1, info_dt_i.shape[1], 15)
    row_num = result_df.shape[0]
    summary_df = result_df.fillna('')
    ws1.write(0, 0, '1.最终入模特征', title_Style)
    ws1.write_row(1, 0, summary_df.columns,title_Style)
    fin_vars = summary_df[u'特征简称']
    for ii in range(row_num):
        ws1.write_row(2 + ii, 0, summary_df.loc[ii,],cell_Style)
        var_name = fin_vars[ii]
        if var_name!='intercept':
            start_num = var_dict[var_name] + 1
            link = "internal:'模型各特征分箱明细（自动分箱）'!A%s" % (str(start_num))  # 写超链接
            ws1.write_url(ii + 2, 3, link, string=var_name)
    ws1.set_column(0,0, 30)
    ws1.set_column(1, result_df.shape[1], 15)
    if out_img:
        ws1.write(row_num + 4, 0, '2.入模特征相关性系数矩阵及热力图',title_Style)
        row_num_df = dfData.shape[0]
        summary_df = dfData.fillna('')
        ws1.write_row(row_num + 5, 0, summary_df.columns,title_Style)
        for ii in range(row_num_df):
            ws1.write_row(row_num + 6 + ii, 0, summary_df.loc[ii,],cell_Style)
        ws1.insert_image(row_num + 8+row_num_df, 0, out_img)
        ws1.set_column(1, dfData.shape[1], 15)
    writer.save()

    # print('--------AUC: ' + str(auc) + '---------')
    # print('---------------KS: -----------------')
    # print(ks_df)
    # print('----------------------------------')
    # print(df_gp_cut)

    # draw_roc([[Y, p], [test[tag], p2]],['train','test'])
    return df_score


def cal_woedf_i(df_rs, bin, tag):
    df_rs['good'] = df_rs[tag].map(lambda x: func_good(x))
    df_rs['bad'] = df_rs[tag].map(lambda x: func_bad(x))
    df_gp = df_rs.groupby([bin])['good', 'bad'].sum()
    bad_sum = df_gp['bad'].sum()
    good_sum = df_gp['good'].sum()
    df_gp['pct_default'] = df_gp['bad'] / (df_gp['bad'] + df_gp['good'])
    df_gp = df_gp.sort_values('pct_default')
    df_gp['bad_pct'] = df_gp['bad'] / np.array([bad_sum for _ in range(len(df_gp))])
    df_gp['good_pct'] = df_gp['good'] / np.array([good_sum for _ in range(len(df_gp))])
    df_gp['Woe'] = np.log(df_gp['good_pct'] / df_gp['bad_pct'])
    df_gp['IV'] = (df_gp['good_pct'] - df_gp['bad_pct']) * df_gp['Woe']
    bad_sum = df_gp['bad'].sum()
    good_sum = df_gp['good'].sum()
    bad_cnt = df_gp['bad'].cumsum()
    good_cnt = df_gp['good'].cumsum()
    df_gp['bad_cnt'] = bad_cnt
    df_gp['good_cnt'] = good_cnt
    df_gp['b_c_p'] = df_gp['bad_cnt'] / np.array([bad_sum for _ in range(len(df_gp))])
    df_gp['g_c_p'] = df_gp['good_cnt'] / np.array([good_sum for _ in range(len(df_gp))])
    df_gp['KS'] = df_gp['g_c_p'] - df_gp['b_c_p']
    df_gp['KS'] = df_gp['KS'].map(lambda x: abs(x))
    return df_gp


def prepare_report_vardetail_cal(i, train, dic_all):
    df_i = pd.DataFrame(dic_all[i[:-4]], index=[i[:-4]]).T
    # df_i=pd.DataFrame(dic_woe_all[i[:-4]],index=[i[:-4]]).T

    df_ii = cal_woedf_i(train, i, 'tag')
    df_ii['woe'] = df_ii['Woe'].map(lambda x: str(x)[:10]).astype(float)
    df_ii = df_ii.reset_index(drop=True)
    df_i['bin'] = df_i.index[:]
    df_i = df_i.reset_index(drop=True)
    df_i[i[:-4]] = df_i[i[:-4]].map(lambda x: str(x)[:10]).astype(float)
    df = pd.merge(df_ii, df_i, how='left', left_on='woe', right_on=i[:-4])
    return df


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


def func_tt_vardescrible(train, test,writer, train_cols, tag):
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
            test_woe = df_describle['CROSS']['Woe'].tolist()
            if pd.Series(test_woe).is_monotonic_decreasing or pd.Series(test_woe).is_monotonic_increasing:
                check.append(0)
            else:
                check.append(1)
            # ————————————判定结束————————————
            df_describle.to_excel(writer,'各入模变量段训练_跨时间集分布对比', startrow=num )
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
    ks_sort.to_excel(writer, '各入模变量段训练_跨时间集指标汇总', startrow=0)
    # return ks_sort