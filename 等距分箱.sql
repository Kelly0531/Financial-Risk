#等距分箱#
#处理列：fin_debit_pct_all_0m_1m #
select created_new,factor,count(*) 
from (
select jg.created_new
      ,jg.account
      ,jg.fin_debit_pct_all_0m_1m
      ,case when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00                            and fin_debit_pct_all_0m_1m <  pct_00 + 1.0 * (pct_09 - pct_00)/9 then concat('0.[',cast(pct_00 as int),',',cast((pct_00 + 1.0*(pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 1.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 2.0 * (pct_09 - pct_00)/9 then concat('1.[',cast((pct_00 + 1.0*(pct_09 - pct_00)/9) as int),',',cast((pct_00 + 2.0 * (pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 2.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 3.0 * (pct_09 - pct_00)/9 then concat('2.[',cast((pct_00 + 2.0*(pct_09 - pct_00)/9) as int),',',cast((pct_00 + 3.0 * (pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 3.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 4.0 * (pct_09 - pct_00)/9 then concat('3.[',cast((pct_00 + 3.0*(pct_09 - pct_00)/9) as int),',',cast((pct_00 + 4.0 * (pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 4.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 5.0 * (pct_09 - pct_00)/9 then concat('4.[',cast((pct_00 + 4.0*(pct_09 - pct_00)/9) as int),',',cast((pct_00 + 5.0 * (pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 5.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 6.0 * (pct_09 - pct_00)/9 then concat('5.[',cast((pct_00 + 5.0*(pct_09 - pct_00)/9) as int),',',cast((pct_00 + 6.0 * (pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 6.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 7.0 * (pct_09 - pct_00)/9 then concat('6.[',cast((pct_00 + 6.0*(pct_09 - pct_00)/9) as int),',',cast((pct_00 + 7.0 * (pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 7.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 8.0 * (pct_09 - pct_00)/9 then concat('7.[',cast((pct_00 + 7.0*(pct_09 - pct_00)/9) as int),',',cast((pct_00 + 8.0 * (pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 8.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 9.0 * (pct_09 - pct_00)/9 then concat('8.[',cast((pct_00 + 8.0*(pct_09 - pct_00)/9) as int),',',cast((pct_00 + 9.0 * (pct_09 - pct_00)/9) as int),')')
            when cnt > 10 and pct_10 >  1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 9.0*(pct_09 - pct_00)/9                                                                    then concat('9.[',cast((pct_00 + 9.0*(pct_09 - pct_00)/9) as int),',',cast(pct_10 as int),']')

            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00                            and fin_debit_pct_all_0m_1m <  pct_00 + 1.0 * (pct_09 - pct_00)/9 then concat('0.[',round(pct_00,2),',',round((pct_00 + 1.0*(pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 1.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 2.0 * (pct_09 - pct_00)/9 then concat('1.[',round((pct_00 + 1.0*(pct_09 - pct_00)/9),2),',',round((pct_00 + 2.0 * (pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 2.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 3.0 * (pct_09 - pct_00)/9 then concat('2.[',round((pct_00 + 2.0*(pct_09 - pct_00)/9),2),',',round((pct_00 + 3.0 * (pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 3.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 4.0 * (pct_09 - pct_00)/9 then concat('3.[',round((pct_00 + 3.0*(pct_09 - pct_00)/9),2),',',round((pct_00 + 4.0 * (pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 4.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 5.0 * (pct_09 - pct_00)/9 then concat('4.[',round((pct_00 + 4.0*(pct_09 - pct_00)/9),2),',',round((pct_00 + 5.0 * (pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 5.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 6.0 * (pct_09 - pct_00)/9 then concat('5.[',round((pct_00 + 5.0*(pct_09 - pct_00)/9),2),',',round((pct_00 + 6.0 * (pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 6.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 7.0 * (pct_09 - pct_00)/9 then concat('6.[',round((pct_00 + 6.0*(pct_09 - pct_00)/9),2),',',round((pct_00 + 7.0 * (pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 7.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 8.0 * (pct_09 - pct_00)/9 then concat('7.[',round((pct_00 + 7.0*(pct_09 - pct_00)/9),2),',',round((pct_00 + 8.0 * (pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 8.0*(pct_09 - pct_00)/9  and fin_debit_pct_all_0m_1m <  pct_00 + 9.0 * (pct_09 - pct_00)/9 then concat('8.[',round((pct_00 + 8.0*(pct_09 - pct_00)/9),2),',',round((pct_00 + 9.0 * (pct_09 - pct_00)/9),2),')')
            when cnt > 10 and pct_10 <= 1.0 and fin_debit_pct_all_0m_1m >= pct_00 + 9.0*(pct_09 - pct_00)/9                                                                    then concat('9.[',round((pct_00 + 9.0*(pct_09 - pct_00)/9),2),',',round(pct_10,2),']')
            else fin_debit_pct_all_0m_1m 
        end as factor 
      ,a.*
from jxq_jg_json_new jg  
left join (
select count(distinct fin_debit_pct_all_0m_1m) as cnt
      ,percentile(fin_debit_pct_all_0m_1m * 100,0.0)/100 as pct_00
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