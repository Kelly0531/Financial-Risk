# -*- coding:utf-8 -*-
param_dic = {
    #系统python路径：
    'system_python_path':'D:\ProgramData\Anaconda3\python.exe',
    ###############d01 特征探查及分布###############
    # d01_项目名
    'project_name': 'data_fraud',
    # d01_数据文件名
    'data_file': 'zz_bcard_sample_merge_2_from1016.csv',
    # d01_Y字段名
    'tag': 'y_flag',
    # d01_测试集比例
    'test_size': 0.2,
    # d01_主键
    'key': 'loan_apply_no',
    # d01_数据分位数
    'percentiles': [0.001, 0.010, 0.025, 0.050, 0.100, 0.200, 0.300, 0.400, 0.500, 0.600, 0.700, 0.800, 0.900, 0.950,
                    0.975, 0.990, 0.999],
    # d01_是否画箱线图
    'bin_image': False,
    ###############d02 特征分箱##############
    # d02_分箱是否添加相关性标签：
    'bin_ex_corr': False,
    # d02_是否输出自动分箱：
    'output_auto_cut_bin': True,
    # d02_是否输出不分箱：
    'output_uncut_bin': False,
    # d02_变量相关性系数阀值
    'corr_limit': 0.8,
    # d02_自动分箱最大箱数
    'auto_bin_num': 5,
    # d02_分箱是否单调
    'mono': True,
    ###############d03 特征衍生##############
    # d03_分箱是否添加相关性标签：
    'file_input': '20190327184447__za_jr_dw_dev_hvs5123_preA_basic_info_with_city_level.csv',
    # d03_输入衍生特征list
    'derivate_var_list': ['age', 'usermobileonlinedd', 'certi_city_level', 'phone_city_level', ],
    # d03_需衍生计算指标
    'derivate_agg_list': ['sum', 'min', 'max', 'mean', 'minus', 'rate', 'std'],
    # 举例两两组合衍生即n=2
    'derivate_var_n': 2,
    #d03_输出衍生变量数据集路径
    'output_data_file': './data/data_derivated.csv',
    # d03_输出衍生变量中文名路径
    'output_datasource_file': './data/data_source_derivated.csv',
    ###############d04 模型训练##############
    # d04模型训练步骤（1：初步训练 2：训练调优 3 跨时间集验证 4跨时间集单变量PSI）
    'step': 1,
    # d04_模型是否去相关性：
    'model_ex_corr': True,
    # d04_评分卡刻度-初始分（训练集整体坏好比）分数
    'base_score': 500,
    # d04_评分卡刻度-训练集整体坏好比的翻倍分数
    'double_score': -50,
    # d04_剔除变量名(默认放入Y标签和主键)
    'ex_cols': ['y_flag', 'loan_apply_no', 'certi_no_md5'],
    # d04_入模必要变量名
    'nece_var': [
                 ],
    # d04_是否仅必要变量名入模
    'is_all_necess': False,
    # d04_跨时间数据文件名，默认放置data文件夹下
    'cross_sample': 'zz_bcard_sample_merge_2_overtime.csv',
    #d04_变量跨时间psi表现
    'model_vars_psi':[]
}
