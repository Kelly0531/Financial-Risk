# -*- coding:utf-8 -*-
import pandas as pd
import numpy as np
import d00_param
import imp
import tool

imp.reload(d00_param)
imp.reload(tool)
import matplotlib.pyplot as plt
from pylab import mpl

mpl.rcParams['font.sans-serif'] = ['SimHei']  # 指定默认字体
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def data_split(df_, tag, test_size, oversample_num=1):
    from sklearn import model_selection
    if float(test_size) == 0.0:
        df_train = df_
        df_test = df_.drop(df_.index)
    else:
        df_y = df_[tag]
        df_x = df_.drop(tag, axis=1)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(
            df_x, df_y, test_size=test_size, random_state=128)
        df_train = pd.concat([x_train, y_train], axis=1)
        df_test = pd.concat([x_test, y_test], axis=1)
    return df_train, df_test


def describle_quantile(data, ex_cols):
    nd_var = list(set(data.columns.tolist()) - set(ex_cols))
    describle = pd.DataFrame()
    num_i = 0
    for k in nd_var:
        try:
            percent = []
            a = data[k]
            # -2填充空值
            a_1 = a.fillna(-2)
            # print('has filled ', k)
            list_a = np.sort(a_1.tolist())
            for i in range(0, 100, 10):
                percent.append(np.percentile(list_a, i))
            describle[k] = percent
            num_i += 1
        except:
            pass
    print("已输入%d 列数据" % num_i)
    describle['postition'] = [str(i * 10) + '%' for i in range(10)]
    describle.set_index(["postition"], inplace=True)
    return describle.T


def get_bin_image(fin_data, data, sheet1, sheet2):
    import os
    if os.path.exists('./image'):
        pass
    else:
        os.makedirs('./image')
    row = 0
    var_list = data.columns
    # var_list=['td_mobile_loan_all_orgcnt_30d']
    var_name = list(fin_data.iloc[:, 0])
    data_source = pd.read_excel('./data/data_source.xlsx', sheet_name='all')
    string_var = data_source[data_source['var_type'] == 'str']['var_name'].tolist()
    for num_ii in var_list:
        if num_ii not in string_var:
            plt.figure()
            # plt.rcParams['font.sans-serif'] = ['SimHei']
            # plt.rcParams['font.family'] = 'serif'
            # plt.rcParams['axes.unicode_minus'] = False
            image_file = './image/' + num_ii + '.png'
            bp = data.boxplot(column=num_ii, sym='r*', meanline=False, showmeans=True, return_type='dict')
            for box in bp['boxes']:
                # change outline color
                box.set(color='#7570b3', linewidth=2)
            ## change color and linewidth of the whiskers
            for whisker in bp['whiskers']:
                whisker.set(color='y', linewidth=2)
            ## change color and linewidth of the caps
            for cap in bp['caps']:  # 上下的帽子
                cap.set(color='#7570b3', linewidth=2)
            ## change color and linewidth of the medians
            for median in bp['medians']:  # 中值
                median.set(color='r', linewidth=2)
            ## change the style of fliers and their fill
            for flier in bp['fliers']:  # 异常值
                flier.set(marker='o', color='k', alpha=0.5)
            plt.savefig(image_file)
            startrow = 3 + 25 * row
            sheet2.write_string(startrow - 1, 0, num_ii)
            sheet2.insert_image(startrow, 0, image_file)  # 插入图片
            start_num = var_name.index(num_ii) + 1
            link = "internal:'箱线图'!A%s" % (str(startrow))  # 写超链接
            sheet1.write_url(start_num, 0, link, string=num_ii)
            row += 1
            plt.close()


def rank(data_i, var_name):
    data_col = data_i[var_name].value_counts()
    data_col = data_col.reset_index()
    data_col['rnk'] = data_col[var_name].rank(ascending=0, method='dense')
    mode1 = ','.join([str(ii) for ii in data_col[data_col['rnk'] == 1]['index'].tolist()])
    mode2 = ','.join([str(ii) for ii in data_col[data_col['rnk'] == 2]['index'].tolist()])
    mode3 = ','.join([str(ii) for ii in data_col[data_col['rnk'] == 3]['index'].tolist()])
    mode4 = ','.join([str(ii) for ii in data_col[data_col['rnk'] == 4]['index'].tolist()])
    mode5 = ','.join([str(ii) for ii in data_col[data_col['rnk'] == 5]['index'].tolist()])
    return [mode1, mode2, mode3, mode4, mode5]


def get_summary_data(newdata):
    all_df = newdata.count()
    all_df = all_df.reset_index()
    all_df.columns = ['var_name', 'count']
    all_df['rate'] = all_df['count'] / key_num
    desc_df = newdata.describe(percentiles=percent_list).T
    desc_df = desc_df.reset_index()
    new_col = desc_df.columns
    new_col = ['var_name' if i == 'index' else i for i in new_col]
    desc_df.columns = new_col
    fin_data = desc_df.copy()
    skew = newdata.skew().reset_index()
    skew.columns = ['var_name', 'skew']
    kurt = newdata.kurt().reset_index()
    kurt.columns = ['var_name', 'kurt']
    fin_data = pd.merge(fin_data, skew, on='var_name', how='left')
    fin_data = pd.merge(fin_data, kurt, on='var_name', how='left')
    # fin_data['skew'] = list()
    # fin_data['kurt'] = list()
    fin_data['mode1'] = fin_data['var_name'].apply(lambda x: rank(newdata, x)[0])
    fin_data['mode2'] = fin_data['var_name'].apply(lambda x: rank(newdata, x)[1])
    fin_data['mode3'] = fin_data['var_name'].apply(lambda x: rank(newdata, x)[2])
    fin_data['mode4'] = fin_data['var_name'].apply(lambda x: rank(newdata, x)[3])
    fin_data['mode5'] = fin_data['var_name'].apply(lambda x: rank(newdata, x)[4])
    new_col = fin_data.columns
    col1 = [i for i in new_col if i.find('%') < 0]
    col2 = [i for i in new_col if i.find('%') >= 0]
    fin_p1 = fin_data[col1].copy()
    # fin_p1['rate'] = fin_p1['count'] / key_num
    fin_p2 = fin_data[col2].copy()
    fin_data = pd.concat([fin_p1, fin_p2], axis=1)
    fin_data = fin_data.drop(['count'], axis=1)
    fin_data = pd.merge(all_df, fin_data, how='left', on='var_name')
    fin_data.sort_values(by='rate', ascending=False, inplace=True)
    fin_data.columns = ['特征中文名', '非空值计数', '覆盖率', '均值', '方差', '最小值', '最大值', '偏度', '峰度', '第1众数', '第2众数', '第3众数', '第4众数',
                        '第5众数'] + col2
    return fin_data


def write_xlsx():
    writer = pd.ExcelWriter(path_ + '_x_describe_info.xlsx')
    # 特征分布情况
    sheet1 = writer.book.add_worksheet('特征探查分布情况')
    sheet1.freeze_panes(1, 1)
    row_num, col_num = fin_data.shape
    title_Style = writer.book.add_format(tool.title_Style)
    cell_Style = writer.book.add_format(tool.cell_Style)
    for jj in range(col_num):
        val = fin_data.columns[jj]
        sheet1.write(0, jj, val, title_Style)
    for ii in range(row_num):
        for jj in range(col_num):
            val = fin_data.iloc[ii, jj]
            if pd.isnull(val):
                val = ''
            if jj == 0:
                new_dict = tool.cell_Style
                new_dict['align'] = 'left'
                fin_cell_Style = writer.book.add_format(new_dict)
            elif jj == 1:
                new_dict = tool.cell_Style
                new_dict['num_format'] = '_ * #,##0_ ;_ * -#,##0_ ;_ * "-"??_ ;_ @_ '
                fin_cell_Style = writer.book.add_format(new_dict)
            elif jj == 2:
                new_dict = tool.cell_Style
                new_dict['num_format'] = '0.0%'
                new_dict['align'] = 'center'
                fin_cell_Style = writer.book.add_format(new_dict)
            else:
                fin_cell_Style = cell_Style
            try:
                sheet1.write(ii + 1, jj, val, fin_cell_Style)
            except:
                sheet1.write(ii + 1, jj, str(val), fin_cell_Style)
    sheet1.set_column(0, 0, 20)
    sheet1.set_column(1, fin_data.shape[1], 15)
    # fin_data.to_excel(writer, 'summary',index=False)
    # 箱线图
    if bin_image:
        sheet2 = writer.book.add_worksheet('箱线图')
        get_bin_image(fin_data, newdata, sheet1, sheet2)
    writer.save()
    writer.close()


def var_type_format(x):
    if (str(x).find('int') >= 0 or str(x).find('float') >= 0):
        return 'number'
    elif str(x) == 'object':
        return 'str'
    else:
        return str(x)


def write_data_resource(df_i, auto_bin_num):
    df_i_old = pd.DataFrame(
        columns=['var_name', 'var_type', 'chn_meaning', 'is_derivated', 'cut_bin_way', 'cut_bin_num'])
    if os.path.exists('./data/data_source.xlsx'):
        df_i_old = pd.read_excel('./data/data_source.xlsx', sheet_name='all')
    df_i_new = df_i.dtypes
    df_i_new = df_i_new.reset_index()
    df_i_new.columns = ['var_name', 'var_type']
    df_i_new = df_i_new.copy()
    df_i_new['var_type'] = df_i_new['var_type'].apply(var_type_format)
    df_i_new.sort_values(by='var_type', ascending=False, inplace=True)
    df_i_new['chn_meaning'] = np.nan
    df_i_new['is_derivated'] = 0
    df_i_new['cut_bin_way'] = 'best_ks'
    df_i_new['cut_bin_num'] = auto_bin_num
    df_i_new = df_i_new[-df_i_new['var_name'].isin(df_i_old['var_name'])]
    df_i_new = pd.concat([df_i_old, df_i_new], axis=0)
    df_i_new.to_excel('./data/data_source.xlsx', sheet_name='all', index=False)
    print(10 * '*' + '请完善字段中文名、数据类型文件（路径为./data/data_source.xlsx）如已完成请忽略' + 10 * '*')


def d01_main(save_path, data, tag, test_size):
    print(10 * '*' + 'start to split data ...' + 10 * '*')
    df_train, df_test = data_split(data, tag, test_size)
    print(10 * '*' + 'finish split data ...' + 10 * '*')
    print(10 * '*' + 'start to describle quantile' + 10 * '*')
    write_xlsx()
    df_train.to_pickle('./data/' + save_path + '_df_train.pkl')
    df_test.to_pickle('./data/' + save_path + '_df_test.pkl')
    print(10 * '*' + 'finish describle quantile' + 10 * '*')


if __name__ == "__main__":
    with tool.Timer() as t:
        param_dic = d00_param.param_dic
        import os

        if os.path.exists('./data'):
            pass
        else:
            os.makedirs('./data')
        # 将数据存放在创建的运行目录下的data文件夹内
        file_in = './data/' + param_dic['data_file']
        percent_list = param_dic['percentiles']
        project_name = param_dic['project_name']
        tag = param_dic['tag']
        key = param_dic['key']
        bin_image = param_dic['bin_image']
        test_size = param_dic['test_size']
        ex_cols = param_dic['ex_cols']
        auto_bin_num = param_dic['auto_bin_num']
        data = tool.read_file(file_in)
        path_ = tool.prepare_path(project_name, 'd01_')
        data = data[data[tag] != 2]
        write_data_resource(data, auto_bin_num)
        key_num = data[key].count()  # 总样本数
        data_col = list(set(data.columns.tolist()) - set([key]))
        newdata = data[data_col]
        fin_data = get_summary_data(newdata)
        d01_main(project_name, data, tag, test_size)
    print(t.secs)
