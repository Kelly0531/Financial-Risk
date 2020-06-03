#等频分箱#
#处理列：fin_debit_pct_all_0m_1m #
select created_new,factor,count(*) 
from (
select jg.created_new
      ,jg.account
      ,jg.fin_debit_pct_all_0m_1m
      ,case when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 and fin_debit_pct_all_0m_1m <  pct_01 then concat('0.[',cast(pct_00 as int),',',cast(pct_01 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_01 and fin_debit_pct_all_0m_1m <  pct_02 then concat('1.[',cast(pct_01 as int),',',cast(pct_02 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_02 and fin_debit_pct_all_0m_1m <  pct_03 then concat('2.[',cast(pct_02 as int),',',cast(pct_03 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_03 and fin_debit_pct_all_0m_1m <  pct_04 then concat('3.[',cast(pct_03 as int),',',cast(pct_04 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_04 and fin_debit_pct_all_0m_1m <  pct_05 then concat('4.[',cast(pct_04 as int),',',cast(pct_05 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_05 and fin_debit_pct_all_0m_1m <  pct_06 then concat('5.[',cast(pct_05 as int),',',cast(pct_06 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_06 and fin_debit_pct_all_0m_1m <  pct_07 then concat('6.[',cast(pct_06 as int),',',cast(pct_07 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_07 and fin_debit_pct_all_0m_1m <  pct_08 then concat('7.[',cast(pct_07 as int),',',cast(pct_08 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_08 and fin_debit_pct_all_0m_1m <  pct_09 then concat('8.[',cast(pct_08 as int),',',cast(pct_09 as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_09 and fin_debit_pct_all_0m_1m <= pct_10 then concat('9.[',cast(pct_09 as int),',',cast(pct_10 as int),']')
            
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 and fin_debit_pct_all_0m_1m <  pct_01 then concat('0.[',cast(round(pct_00,2) as string),',',cast(round(pct_01,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_01 and fin_debit_pct_all_0m_1m <  pct_02 then concat('1.[',cast(round(pct_01,2) as string),',',cast(round(pct_02,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_02 and fin_debit_pct_all_0m_1m <  pct_03 then concat('2.[',cast(round(pct_02,2) as string),',',cast(round(pct_03,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_03 and fin_debit_pct_all_0m_1m <  pct_04 then concat('3.[',cast(round(pct_03,2) as string),',',cast(round(pct_04,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_04 and fin_debit_pct_all_0m_1m <  pct_05 then concat('4.[',cast(round(pct_04,2) as string),',',cast(round(pct_05,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_05 and fin_debit_pct_all_0m_1m <  pct_06 then concat('5.[',cast(round(pct_05,2) as string),',',cast(round(pct_06,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_06 and fin_debit_pct_all_0m_1m <  pct_07 then concat('6.[',cast(round(pct_06,2) as string),',',cast(round(pct_07,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_07 and fin_debit_pct_all_0m_1m <  pct_08 then concat('7.[',cast(round(pct_07,2) as string),',',cast(round(pct_08,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_08 and fin_debit_pct_all_0m_1m <  pct_09 then concat('8.[',cast(round(pct_08,2) as string),',',cast(round(pct_09,2) as string),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_09 and fin_debit_pct_all_0m_1m <= pct_10 then concat('9.[',cast(round(pct_09,2) as string),',',cast(round(pct_10,2) as string),']')
            else fin_debit_pct_all_0m_1m 
        end as factor 
      ,a.*
from jxq_jg_json_new jg  
left join (
select count(distinct fin_debit_pct_all_0m_1m) as cnt
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.0)/100 as pct_00
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.1)/100 as pct_01
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.2)/100 as pct_02
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.3)/100 as pct_03
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.4)/100 as pct_04
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.5)/100 as pct_05
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.6)/100 as pct_06
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.7)/100 as pct_07
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.8)/100 as pct_08
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.9)/100 as pct_09
      ,percentile(fin_debit_pct_all_0m_1m * 100,1.0)/100 as pct_10
from jxq_jg_json_new 
where fin_debit_pct_all_0m_1m is not null 
)a on 1 = 1
where jg.fin_debit_pct_all_0m_1m is not null 
)aa 
group by created_new,factor 
order by created_new,factor 
limit 999;