# -*- coding:utf-8 -*-
import os
import d00_param
import imp

imp.reload(d00_param)
task_dict = {1: "d01_read_data.py", 2: "d02_woe_bin.py", 3: "d03_var_derivation.py", 4: "d04_modeling.py"}


def ExecuteFunction(py_path,task_list):
    if task_list:
        if 1 in task_list:
            run_file = task_dict[1]
            os.system(py_path + " " + run_file)
        if 2 in task_list:
            run_file = task_dict[2]
            os.system(py_path + " " + run_file)
        if 3 in task_list:
            run_file = task_dict[3]
            os.system(py_path + " " + run_file)
        if 4 in task_list:
            run_file = task_dict[4]
            os.system(py_path + " " + run_file)


if __name__ == "__main__":
    # python系统路径
    param_dic = d00_param.param_dic
    system_python_path = param_dic['system_python_path']
    # 执行文件列表 1: "d01_read_data.py", 2: "d02_woe_bin.py", 3: "d03_var_derivation.py", 4: "d04_modeling.py"
    task_list = [1,2,4]
    ExecuteFunction(system_python_path, task_list)
