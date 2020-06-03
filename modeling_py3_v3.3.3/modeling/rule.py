import pandas as pd
import tool

def create_rule(data,rule_data,rand_num=[]):
    var_list=list(rule_data['var_name'].unqiue())
    if len(vars)==0:
        print('error')
    else:
        for num in rand_num:
            new_list=tool.combine(rule_data,num)


def ai_rule_test():
    df=pd.read_csv('./data.csv')
    df['is_touch']=((df['var1']>10)&(df['var2']>10)|(df['var3']>10))
    sample_num=df.shape[0]
    var_name='is_touch'
    y_flag='y2_new'
    df_i=df[[var_name,y_flag]].fillna('Null')
    new=df_i[y_flag].groupby([df_i[var_name],df_i[y_flag]]).count().unstack().reset_index().fillna(0)
    new.columns=['var_value','good','bad']
    new['bin_num']=new['good']+new['bad']
    new['bad_rate']=new['bad']/new['total']
    new['bin_rate']=new['bin_num']/sample_num
    print(new)