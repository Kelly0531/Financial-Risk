# -*- coding: utf-8 -*-


# import sqlalchemy as sql
import pandas     as pd
import numpy      as np
import xlsxwriter as xw
import xlrd       as xl
import sys
import os
import datetime


def draw_woe(info,var_list,save_path):
    today = datetime.date.today().strftime('%Y%m%d')
    workbook = xw.Workbook("{1}_vars_woe_{0}.xlsx".format(today,save_path))
    sheet_w = workbook.add_worksheet('Detail')
    Title_fmt = workbook.add_format(
        {'font_name': 'Segoe UI', 'font_color': 'white', 'font_size': 11, 'bold': True, 'pattern': 1,
         'bg_color': '#0f243e', 'border': 1, 'border_color': 'white'})
    title_fmt = workbook.add_format(
        {'font_name': 'Segoe UI',
         'font_color': 'black',
         'font_size': 11,
         'bold': True,
         'pattern': 1,
         'bg_color': '#cccccc',
         'border': 1,
         'border_color': 'black'})
    data_fmt = workbook.add_format(
        {'font_name': 'Segoe UI',
         'font_color': 'black',
         'font_size': 9,
         'bold': False,
         'pattern': 1,
         'bg_color': '#cccccc',
         'border': 1,
         'align':'center',
         'border_color': 'black',
         'num_format': '0.00'})
    data_fmt2 = workbook.add_format(
        {'font_name': 'Segoe UI',
         'font_color': 'black',
         'font_size': 9,
         'bold': False,
         'pattern': 1,
         'bg_color': '#cccccc',
         'border': 1,
         'align':'center',
         'border_color': 'black',
         'num_format': '0.00%'})
    curr_row = 0
    for var_names in var_list:
        print (var_names)
        if var_names in info['var_name'].tolist():
            data=info[info['var_name']==var_names]
            var_name=data.columns[0]
            tmp = data[[var_name,'Total_Num','Bad_Num','Good_Num','Total_Pcnt','Bad_Rate','Woe','IV','KS']]
            tmp=tmp.fillna(0)
            columns=[var_names,'Total_Num','Bad_Num','Good_Num','Total_Pcnt','Bad_Rate','Woe','IV','KS']
            tmp_o=tmp[tmp[var_name]!='(-1.0,-1.0)'].reset_index(drop=True)
            tmp_1=None
            # write to the xlsx
            sheet_w.write_string(curr_row, 0, var_names, Title_fmt)
            curr_row += 1
            # print columns
            for col in range(len(columns)):
                sheet_w.write_string(curr_row, col, columns[col], title_fmt)
            if '(-1.0,-1.0)' in tmp[var_name].tolist():
                tmp_1 = tmp[tmp[var_name] == '(-1.0,-1.0)'].reset_index(drop=True).iloc[0].tolist()
                curr_row += 1
                for col in range(len(columns)):
                    if col==0:
                        sheet_w.write_string(curr_row, col, str(tmp_1[col]), data_fmt)
                    else:
                        sheet_w.write_number(curr_row, col, tmp_1[col], data_fmt)
            curr_row += 1
            first_row = curr_row + 1
            for row in range(tmp_o.shape[0]):
                        #var_name default index = 0
                sheet_w.write_string(curr_row, 0, tmp_o[var_name][row], data_fmt)
                sheet_w.write_number(curr_row, 1, tmp_o['Total_Num'][row], data_fmt)
                sheet_w.write_number(curr_row, 2, tmp_o['Bad_Num'][row], data_fmt)
                sheet_w.write_number(curr_row, 3, tmp_o['Good_Num'][row], data_fmt)
                sheet_w.write_number(curr_row, 4, tmp_o['Total_Pcnt'][row], data_fmt2)
                sheet_w.write_number(curr_row, 5, tmp_o['Bad_Rate'][row], data_fmt2)
                sheet_w.write_number(curr_row, 6, tmp_o['Woe'][row], data_fmt)
                sheet_w.write_number(curr_row, 7, tmp_o['IV'][row], data_fmt)
                sheet_w.write_number(curr_row, 8, tmp_o['KS'][row], data_fmt2)
                curr_row+=1
            end_row = curr_row
            chart = workbook.add_chart({'type': 'bar'})
            chart.set_size({'height': 200, 'width': 600})
            if tmp_1:
                begin_row=first_row-1
            else:
                begin_row=first_row
            chart.add_series({
                'values': '={2}!$G${0}:$G${1}'.format(begin_row, end_row, 'Detail'),
                'categories':'={2}!$A${0}:$A${1}'.format(begin_row, end_row, 'Detail'),
                # 'data_labels': {'value': True, 'position': 'outside_end'},
                'name': '',
                'trendline':{
                    'type':'polynomial',
                    'order':3,
                    'name':'trend'
                }
            })
            chart.set_x_axis({'name':var_names+'_woe'})
            chart.set_y_axis({'name':'bin'})
            chart.set_style(1)
            # chart.set_chartarea({'fill':{
            #     'color':'blue',
            #     'transpatency':70
            # }})
            sheet_w.insert_chart(first_row-3, 11, chart)
            curr_row += 4
    workbook.close()

# draw_woe(info,train_cols,'TD')


def draw_woe_fromdic(data_dic,save_path):
    today = datetime.date.today().strftime('%Y%m%d')
    workbook = xw.Workbook("{1}_vars_woe_{0}.xlsx".format(today,save_path))
    sheet_w = workbook.add_worksheet('Detail')
    Title_fmt = workbook.add_format(
        {'font_name': 'Segoe UI', 'font_color': 'white', 'font_size': 11, 'bold': True, 'pattern': 1,
         'bg_color': '#0f243e', 'border': 1, 'border_color': 'white'})
    title_fmt = workbook.add_format(
        {'font_name': 'Segoe UI',
         'font_color': 'black',
         'font_size': 11,
         'bold': True,
         'pattern': 1,
         'bg_color': '#cccccc',
         'border': 1,
         'border_color': 'black'})
    data_fmt = workbook.add_format(
        {'font_name': 'Segoe UI',
         'font_color': 'black',
         'font_size': 9,
         'bold': False,
         'pattern': 1,
         'bg_color': '#cccccc',
         'border': 1,
         'align':'center',
         'border_color': 'black',
         'num_format': '0.00'})
    data_fmt2 = workbook.add_format(
        {'font_name': 'Segoe UI',
         'font_color': 'black',
         'font_size': 9,
         'bold': False,
         'pattern': 1,
         'bg_color': '#cccccc',
         'border': 1,
         'align':'center',
         'border_color': 'black',
         'num_format': '0.00%'})
    curr_row = 0
    for var_name,data in data_dic.items():

        # print (var_name)
        if data.shape[0]>0:
            try:
               # data=data.reset_index()
               tmp = data[[var_name,'Total_Num','Bad_Num','Good_Num','Total_Pcnt','Bad_Rate','Woe','IV','KS']]
               tmp=tmp.fillna(0)
               columns=[var_name,'Total_Num','Bad_Num','Good_Num','Total_Pcnt','Bad_Rate','Woe','IV','KS']
               tmp_o=tmp[tmp[var_name]!='(-1.0,-1.0)'].reset_index(drop=True)
               tmp_1=None
               # write to the xlsx
               sheet_w.write_string(curr_row, 0, var_name, Title_fmt)
               curr_row += 1
               # print columns
               for col in range(len(columns)):
                   sheet_w.write_string(curr_row, col, columns[col], title_fmt)
               if '(-1.0,-1.0)' in tmp[var_name].tolist():
                   tmp_1 = tmp[tmp[var_name] == '(-1.0,-1.0)'].reset_index(drop=True).iloc[0].tolist()
                   curr_row += 1
                   for col in range(len(columns)):
                       if col==0:
                           sheet_w.write_string(curr_row, col, str(tmp_1[col]), data_fmt)
                       else:
                           sheet_w.write_number(curr_row, col, tmp_1[col], data_fmt)
               curr_row += 1
               first_row = curr_row + 1
               for row in range(tmp_o.shape[0]):
                           #var_name default index = 0
                   sheet_w.write_string(curr_row, 0, tmp_o[var_name][row], data_fmt)
                   sheet_w.write_number(curr_row, 1, tmp_o['Total_Num'][row], data_fmt)
                   sheet_w.write_number(curr_row, 2, tmp_o['Bad_Num'][row], data_fmt)
                   sheet_w.write_number(curr_row, 3, tmp_o['Good_Num'][row], data_fmt)
                   sheet_w.write_number(curr_row, 4, tmp_o['Total_Pcnt'][row], data_fmt2)
                   sheet_w.write_number(curr_row, 5, tmp_o['Bad_Rate'][row], data_fmt2)
                   sheet_w.write_number(curr_row, 6, tmp_o['Woe'][row], data_fmt)
                   sheet_w.write_number(curr_row, 7, tmp_o['IV'][row], data_fmt)
                   sheet_w.write_number(curr_row, 8, tmp_o['KS'][row], data_fmt2)
                   curr_row+=1
               end_row = curr_row
               chart = workbook.add_chart({'type': 'bar'})
               chart.set_size({'height': 200, 'width': 600})
               if tmp_1:
                   begin_row=first_row-1
               else:
                   begin_row=first_row
               chart.add_series({
                   'values': '={2}!$G${0}:$G${1}'.format(begin_row, end_row, 'Detail'),
                   'categories':'={2}!$A${0}:$A${1}'.format(begin_row, end_row, 'Detail'),
                   # 'data_labels': {'value': True, 'position': 'outside_end'},
                   'name': '',
                   'trendline':{
                       'type':'polynomial',
                       'order':3,
                       'name':'trend'
                   }
               })
               chart.set_x_axis({'name':var_name+'_woe'})
               chart.set_y_axis({'name':'bin'})
               chart.set_style(1)
               # chart.set_chartarea({'fill':{
               #     'color':'blue',
               #     'transpatency':70
               # }})
               sheet_w.insert_chart(first_row-3, 11, chart)
               curr_row += 4
            except:
                print (var_name)
                print(data)
    workbook.close()