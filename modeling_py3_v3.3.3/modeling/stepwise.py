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
# from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from modeling.cal_ks_auc_score import func_good,func_bad,cal_ks
from sklearn.utils import shuffle


def func_sort_col(col, df_, code):
    '''
    该函数用于将train_cols按wald_chi2排序：
    ：wald_chi2= (coef/se(coef))^2
    ：se(coef)=sd(coef)/n^2
    :param col:trian_cols[list]
    :param df_: m_data[pd.dataframe]
    :param code: tag_name
    :return: sorted_cols[list]
    '''
    cols = col[:]
    dic_wald = {}
    # for i in cols:
    #     dic_wald[i]=cal_wald_chi2(i,df_,code)
    #
    # wald_df=pd.DataFrame(dic_wald,index=['wald']).T
    # sorted_cols=list(wald_df.sort_values('wald',ascending=False).index)
    if 'intercept' not in cols:
        cols.append('intercept')
        df_['intercept'] = 1
    cols = list(set(cols) & set(df_.columns))
    logit_1 = sm.Logit(df_[code], df_[cols])
    result_1 = logit_1.fit()
    wald_chi2 = np.square((result_1.params) / np.square(result_1.bse))
    a = pd.DataFrame(wald_chi2, columns=['value'])
    b = a.sort_values('value', ascending=False)
    sorted_cols = b.index.tolist()
    sorted_cols.remove('intercept')
    return a, sorted_cols


#
# def cal_wald_chi2(var,df_,code):
#     try:
#         df_['intercept']=1
#         logit_i=sm.Logit(df_[code],df_[var])
#         result_1=logit_i.fit()
#         wald_chi2 = np.square((result_1.params) / np.square(result_1.bse))
#     except Exception,E:
#         print Exception,E
#         wald_chi2=0
#     return wald_chi2


# sorted_cols=func_sort_col(vars_,y_train,'tag7')

# def cal_ks_gp(df_rs,bin,tag):
#     df_rs.ix[:,'good']=df_rs[tag].map(lambda x:func_good(x))
#     df_rs.ix[:,'bad']=df_rs[tag].map(lambda x:func_bad(x))
#     df_gp=df_rs.groupby([bin])['good','bad'].sum()
#     bad_sum=df_gp['bad'].sum()
#     good_sum=df_gp['good'].sum()
#     bad_cnt=df_gp['bad'].cumsum()
#     good_cnt=df_gp['good'].cumsum()
#     df_gp.ix[:,'bad_cnt']=bad_cnt
#     df_gp.ix[:,'good_cnt']=good_cnt
#     df_gp['bad_pct']=df_gp['bad']/np.array([bad_sum for _ in range(len(df_gp))])
#     df_gp['good_pct']=df_gp['good']/np.array([good_sum for _ in range(len(df_gp))])
#     df_gp['Woe']=np.log(df_gp['good_pct']/df_gp['bad_pct'])
#     df_gp['IV'] = (df_gp['good_pct'] - df_gp['bad_pct']) * df_gp['Woe']
#     df_gp.ix[:,'b_c_p']=df_gp['bad_cnt']/np.array([bad_sum for _ in range(len(df_gp))])
#     df_gp.ix[:,'g_c_p']=df_gp['good_cnt']/np.array([good_sum for _ in range(len(df_gp))])
#     df_gp.ix[:,'ks']=df_gp['g_c_p']-df_gp['b_c_p']
#     ks_max=max(df_gp['ks'].map(lambda x :abs(x)))
#     iv=sum(df_gp['IV'].tolist())
#     return ks_max,iv


def func_sort_min_innerks(train_cols, data_path, tag):
    t_data = pd.read_csv('T_Data_' + data_path + '.csv')
    info = pd.read_excel('Info_' + data_path + '.xlsx')
    test_ks = pd.DataFrame()
    varname_list = []
    varks_list = []
    for i in train_cols:
        data_i = t_data[[i[4:], tag, 'id']]
        ks_i = gb_add_woe(data_i, i[4:])
        varname_list.append(i[4:])
        varks_list.append(ks_i)
    test_ks['var'] = varname_list
    test_ks['test_ks'] = varks_list
    ks_df = pd.merge(test_ks, info, how='left', on='var')
    ks_df['ks_dif'] = abs(ks_df['ks'] - ks_df['test_ks'])
    ks_sort = ks_df.sort_values('ks_dif', ascending=True)
    a = ks_sort['var'].tolist()
    b = ['WOE_' + i for i in a]
    print(ks_sort[['ks', 'test_ks', 'ks_dif']])
    return b


# sorted_cols=func_sort_min_innerks(train_cols_cor,'yk_new_allvars','tag_x')
# cols=['WOE_'+i for i in sorted_cols]


def func_check_(x):
    list_1 = list(x)
    check = []
    for i in list_1:
        if i < 0:
            check.append(1)
        else:
            check.append(0)
    if sum(check) == len(list_1):
        return True
    else:
        return False


# ##l2建模步骤
# def func_stepwise_1(cols, df_, code, report=None):
#     '''
#     :param cols: train_cols[list]
#     :param df_: m_data[pd.df]
#     :param code: tag_name
#     :param sort: 是否需要进行排序：TRUE/FALSE
#     :return: 入参变量
#     '''
#     vars_dic = {}
#     import statsmodels.api as sm
#     train_cols = ['intercept']
#     df_['intercept'] = 1
#     error_var = []
#     sorted_cols = cols[:]
#     num = 0
#     if report:
#         lr =LogisticRegression(penalty= 'l2', C= 0.5)
#         X_shuf, Y_shuf = shuffle(df_[cols].as_matrix(), df_[code].as_matrix())
#         result = lr.fit(X_shuf,Y_shuf)
#
#         # result = logit.fit()
#     else:
#         for i in sorted_cols:
#             print ('has looped to ' + str(i))
#             train_cols = list(set(train_cols) | set(['intercept', i]))
#             try:
#                 ori = train_cols[:]
#                 lr = LogisticRegression(penalty='l2', C=0.5)
#                 logit = lr.fit(df_[ori],df_[code])
#                 result = logit.fit()
#                 train_p = result.pvalues[result.pvalues < 0.05].index.tolist()
#                 train_params = result.params[result.params > 0].index.tolist()
#                 train_cols = list(set(train_p) - set(train_params))
#                 # print result.summary()
#                 # if len(ori)==len(train_cols) and len(ori)>15 and len(ori)<30:
#                 #     p = result.predict().tolist()
#                 #     Y = df_[code]
#                 #     auc = roc_auc_score(Y, p)
#                 #     print result.summary()
#                 #     print '===================================='
#                 #     print auc
#                 #     if func_check_(result.params) and auc>0.65:
#                 #         vars_dic[num]=[train_cols,auc]
#                 #         print 'the vars can be modeled',auc
#                 #         print len(ori)-1
#                 #         print ori
#                 #         print 'has saved vars'
#                 # else:
#                 #     continue
#             except Exception:
#                 # print Exception,":",e
#                 print('some error ')
#                 train_cols.remove(i)
#                 error_var.append(i)
#     train_p = result.pvalues[result.pvalues < 0.05].index.tolist()
#     train_params = result.params[result.params > 0].index.tolist()
#     train_cols = list(set(train_p) - set(train_params))
#     lr = LogisticRegression(penalty='l2', C=0.5)
#     logit = lr.fit( df_[train_cols],df_[code])
#     result = logit.fit()
#     print(result.summary2())
#     # print result.summary2()
#     print ('aic:%f' % (float(result.aic)))
#     p = result.predict().tolist()
#     Y = df_[code]
#     auc = roc_auc_score(Y, p)
#     # df_['p']=p
#     ks_df = cal_ks(result, df_, code, 550, 40)
#     # ks_max, df_gp=gb_add_woe(df_,code)
#     # print 'max_ks_uncut:' +str(ks_max)
#     # detail_p=df_[['p',code]]
# return train_cols,result,vars_dic

# return train_cols, result, vars_dic, ks_df.ix[1, 'describe'], error_var

def pmml(x, Y):
    from sklearn2pmml import PMMLPipeline, sklearn2pmml

    LR_pipeline = PMMLPipeline([
        ("classifier", LogisticRegression())
    ])

    # 训练模型
    LR_pipeline.fit(x, Y)
    sklearn2pmml(LR_pipeline, "LogisticRegression.pmml")
    # 导出模型到 RandomForestClassifier_Iris.pmml 文件


# 原建模步骤
def func_stepwise_1(cols, df_, code, report=None, pmml_btn=True,nec_vars=[],base_score=550,double_score=-40):
    '''
    :param cols: train_cols[list]
    :param df_: m_data[pd.df]
    :param code: tag_name
    :param sort: 是否需要进行排序：TRUE/FALSE
    :return: 入参变量
    '''
    vars_dic = {}
    import statsmodels.api as sm
    train_cols = ['intercept']
    df_['intercept'] = 1
    error_var = []
    sorted_cols = cols[:]
    nec_vars = [i + "_woe" for i in nec_vars if i + "_woe" in df_.columns]
    num = 0
    if report:
        logit = sm.Logit(df_[code], df_[cols])
        result = logit.fit()
    else:
        for i in sorted_cols:
            print('has looped to ' + str(i))
            train_cols = list(set(train_cols) | set(['intercept', i]))
            try:
                ori = train_cols[:]
                logit = sm.Logit(df_[code], df_[ori])
                result = logit.fit()
                train_p = result.pvalues[result.pvalues < 0.05].index.tolist()
                train_params = result.params[result.params > 0].index.tolist()
                train_cols = list(set(train_p) - set(train_params))
                # print result.summary()
                # if len(ori)==len(train_cols) and len(ori)>15 and len(ori)<30:
                #     p = result.predict().tolist()
                #     Y = df_[code]
                #     auc = roc_auc_score(Y, p)
                #     print result.summary()
                #     print '===================================='
                #     print auc
                #     if func_check_(result.params) and auc>0.65:
                #         vars_dic[num]=[train_cols,auc]
                #         print 'the vars can be modeled',auc
                #         print len(ori)-1
                #         print ori
                #         print 'has saved vars'
                # else:
                #     continue
            except Exception:
                # print Exception,":",e
                print('some error ')
                train_cols.remove(i)
                error_var.append(i)
    #
    # train_p = result.pvalues[result.pvalues < 0.05].index.tolist()
    # train_params = result.params[result.params > 0].index.tolist()
    train_cols = list(set(train_cols)|set(nec_vars)| set(['intercept']))

    # lr = LogisticRegression(penalty='l2', C=0.5)
    # result = lr.fit(df_[train_cols],df_[code])
    logit = sm.Logit(df_[code], df_[train_cols])
    result = logit.fit()
    if pmml_btn:
        pmml(df_[train_cols], df_[code])
    print(result.summary2())
    print('aic:%f' % (float(result.aic)))
    p = result.predict().tolist()
    # print(p)
    Y = df_[code]
    auc = roc_auc_score(Y, p)
    # df_['p']=p
    ks_df,odds,A,B = cal_ks(result, df_, code, base_score, double_score)
    # ks_max, df_gp=gb_add_woe(df_,code)
    # print 'max_ks_uncut:' +str(ks_max)
    # detail_p=df_[['p',code]]
    # return train_cols,result,vars_dic

    return train_cols, result, vars_dic, ks_df.ix[1, 'describe'], error_var,odds,A,B



def func_stepwise_sk(cols, df_, code):
    '''
    :param cols: train_cols[list]
    :param df_: m_data[pd.df]
    :param code: tag_name
    :param sort: 是否需要进行排序：TRUE/FALSE
    :return: 入参变量
    '''

    vars_dic = []
    from sklearn import linear_model as lm
    lr = lm.LogisticRegression(C=1.0, class_weight=None, dual=False,
                               fit_intercept=True, intercept_scaling=1,
                               max_iter=100, multi_class='ovr', n_jobs=1,
                               penalty='l2', random_state=None,
                               solver='liblinear', tol=0.0001,
                               verbose=0, warm_start=False)
    sorted_cols = cols[:]
    for i in sorted_cols:
        print('has looped to ' + str(i))
        train_cols.append(i)
        try:
            ori = train_cols[:]
            X_shuf, Y_shuf = shuffle(df_[ori].as_matrix(), df_[code].as_matirx())
            logit = lr.fit(X_shuf, Y_shuf)
            result = logit.fit()
            train_p = result.pvalues[result.pvalues < 0.03].index.tolist()
            train_params = result.params[result.params > 0].index.tolist()
            train_cols = list(set(train_p) - set(train_params))
            # print result.summary()
            # if len(ori)==len(train_cols) and len(ori)>15 and len(ori)<25:
            #     p = result.predict().tolist()
            #     Y = df_[code]
            #     auc = roc_auc_score(Y, p)
            #     if func_check_(result.params) and auc>0.72:
            #         vars_dic.append(train_cols)
            #         print '这些变量可以用来建模啦',auc
            #         print len(ori)-1
            #         print ori
            #         print 'has saved vars'
            # else:
            #     continue
        except Exception as  e:
            print(":", e)
            # print 'some error '
            train_cols.remove(i)
    logit = sm.Logit(df_[code], df_[train_cols])
    result = logit.fit()
    print(result.summary())
    # print result.summary2()
    print('aic： ' + str(result.aic))
    p = result.predict().tolist()
    Y = df_[code]
    auc = roc_auc_score(Y, p)
    # df_['p']=p
    ks_df = cal_ks(result, df_, code, 550, 40)
    # ks_max, df_gp=gb_add_woe(df_,code)
    # print 'max_ks_uncut:' +str(ks_max)
    # detail_p=df_[['p',code]]
    # return train_cols,result,vars_dic
    return train_cols, result, vars_dic, ks_df.ix[1, 'describe']


def group_p(data, code):
    df_rs = data.copy()
    list = df_rs.index.tolist()
    df_rs.ix[:, 'no'] = list
    df_rs['p_bin'] = df_rs['p'][:]
    df_rs['good'] = df_rs[code].map(lambda x: func_good(x))
    df_rs['bad'] = df_rs[code].map(lambda x: func_bad(x))
    b = df_rs.groupby(['p_bin', ])['no', 'good', 'bad'].sum()
    df_gp = pd.DataFrame(b)
    return df_gp


def gb_add_woe(data, code):
    df_gp = group_p(data, code)
    df_gp.ix[:, 'pct_default'] = df_gp['bad'] / (df_gp['bad'] + df_gp['good'])
    bad_sum = df_gp['bad'].sum()
    good_sum = df_gp['good'].sum()
    df_gp.ix[:, 'bad_pct'] = df_gp['bad'] / np.array([bad_sum for _ in range(len(df_gp))])
    df_gp.ix[:, 'good_pct'] = df_gp['good'] / np.array([good_sum for _ in range(len(df_gp))])
    df_gp.ix[:, 'odds'] = df_gp['good_pct'] - df_gp['bad_pct']
    df_gp.ix[:, 'woe'] = np.log(df_gp['good_pct'] / df_gp['bad_pct'])
    df_gp.ix[:, 'iv'] = (df_gp['good_pct'] - df_gp['bad_pct']) * df_gp['woe']
    bad_cnt = df_gp['bad'].cumsum()
    good_cnt = df_gp['good'].cumsum()
    df_gp.ix[:, 'bad_cnt'] = bad_cnt
    df_gp.ix[:, 'good_cnt'] = good_cnt
    df_gp.ix[:, 'b_c_p'] = df_gp['bad_cnt'] / np.array([bad_sum for _ in range(len(df_gp))])
    df_gp.ix[:, 'g_c_p'] = df_gp['good_cnt'] / np.array([good_sum for _ in range(len(df_gp))])
    df_gp.ix[:, 'ks'] = df_gp['g_c_p'] - df_gp['b_c_p']
    ks_max = df_gp['ks'].max()
    # print df_gp
    return ks_max, df_gp


def func_stepwise(sorted_cols, df_, code):
    train_cols = ['intercept']
    num = 1
    for j in sorted_cols:
        # print 'has looped to ' + str(j)
        for i in sorted_cols:
            train_cols.append(i)
            train_cols = list(set(train_cols))
            # logit = sm.Logit(df_[train_cols],df_[code])
            logit = sm.Logit(df_[code], df_[train_cols])
            result = logit.fit()
            train_cols = result.pvalues[result.pvalues < 0.05].index.tolist()
        try:
            train_cols.append(j)
            train_cols = list(set(train_cols))
            logit = sm.Logit(df_[code], df_[train_cols])
            result = logit.fit()
            train_cols = result.pvalues[result.pvalues < 0.05].index.tolist()
        except:
            pass
            # print train_cols

    logit = sm.Logit(df_[code], df_[train_cols])
    result = logit.fit()
    # print result.summary()
    # print result.summary2()
    return train_cols


# col=func_stepwise(coll,M_data,'tag')


def forward_selected(data, response):
    """Linear model designed by forward selection.

    Parameters:
    -----------
    data : pandas DataFrame with all possible predictors and response

    response: string, name of response column in data

    Returns:
    --------
    model: an "optimal" fitted statsmodels linear model
           with an intercept
           selected by forward selection
           evaluated by adjusted R-squared
    """
    import statsmodels.formula.api as smf
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = 0.0, 0.0
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {} + 1".format(response,
                                           ' + '.join(selected + [candidate]))
            score = smf.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score < best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {} + 1".format(response,
                                   ' + '.join(selected))
    model = smf.ols(formula, data).fit()
    return model
