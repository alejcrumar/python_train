

*   dl_basel
*	Teradata view control



-- grant access to tables 
GRANT SELECT ON ud160.u91111_rohan_final_month_0_12 TO DZU532

-- create table;	
create multiset table ud501.alejandro_test as (
	select *
	from pcdw.pcan_stmt
	sample 100) with data primary index (acct_id);


-- Extract subset from table ;
create multiset table ud502.mlb198_test as (
select *
from pcdw.acct_cnvrsn
sample 100) with data primary index (app_id);

-- delete table;
drop table ud502.mlb198_test;

-- help tables; 
help table pcdw.acct_cnvrsn;
help column pcdw.acct_cnvrsn.*;

-- statistics ;
help statistics pcdw.acct_cnvrsn;

-- select;
select count(*), count(acct_id) from pcdw.acct_cnvrsn;
select count(*), count(acct_id) from pcdw.acct_score;
select count(*)  from pcdw.acct_score;          
select  top 10  *  from pcdw.acct_score;
select count(*) as count_withnulls, count(acct_id) count_nonulls  from pcdw.pcan_stmt;
select top 10 * from pcdw.pcan_stmt;


-- show;
show view pcdw.acct_score;

-- views;
help view

-- dates;
SELECT EXTRACT (YEAR FROM '2005-01-01'); 
SELECT ADD_MONTHS ('2005-01-01', 1); 
SELECT date/100 

 
-- average;
select avg(FINL_MODEL_SCORE_VAL)
	from pcdw.acct_score
	where FINL_MODEL_SCORE_VAL between 600 and 700;
	
	
-- operations in select ; 
select sum(pmt_amt) 
	from pcdw.pcan_stmt
	where pmt_amt between 100 and 300;

select avg(pmt_amt), min(pmt_amt), max(pmt_amt), count(pmt_amt), count(*)
	from pcdw.pcan_stmt;
	

-- distinct values; 			
select distinct (PMT_AMT_MADE_CD) FROM pcdw.pcan_stmt

-- operations with case ;
select  SUM (CASE WHEN PMT_AMT_MADE_CD ='G' THEN 1 ELSE 0 
 		END) AS total_categorical
	from pcdw.pcan_stmt;	

	
select sum (case when APPLICATION_STATUS = 'A' then 1 else 0 end) as total_a,
	sum (case when APPLICATION_STATUS = 'D' then 1 else 0 end) as total_d,
	sum (case when APPLICATION_STATUS = 'P' then 1 else 0 end)  as total_p
from UD117.u1005c_cash_application
where  SOLICITATION_ID = 8422;

-- group;	
select PMT_AMT_MADE_CD
 	FROM pcdw.pcan_stmt
 	group by PMT_AMT_MADE_CD;


-- duplicates;
select count(PMT_AMT_MADE_CD), count(distinct PMT_AMT_MADE_CD )
	 from pcdw.pcan_stmt;

-- Find unique observations ;
select count(distinct accountnumber) from ud502.u58417_h_mdl_app_mntr
where accountnumber is not NULL;	 
	 
-- Find repeated values. count repetitions  Duplicates
select accountnumber, count(*) 
from ud502.u58417_h_mdl_app_mntr
group by 1
having count(accountnumber)>1;
	 
	 
-- sample;
select *   from pcdw.pcan_stmt sample 0.00000001;
	
create multiset table ud502.mlb198_test as (
	select *
	from padw_fl.fact_cntrt_snap
	sample 100) with data primary index (app_id);


--Determine size of a table ;
select sum(currentperm) 
	from dbc.allspace 
		where databasename = 'ud502' and tablename = 'u58417_h_mdl_app_mntr';


		
-- Select observations that satisfy a given condition ;
select * from udw.apac_monthly_hist where  cm11 in ( 30004706971,  30004706972,  30004706973
-- Select all the observations with repeated values ;
select * from ud502.u58417_h_mdl_app_mntr 
where accountnumber IN (select accountnumber
from ud502.u58417_h_mdl_app_mntr 
group by 1
having count(accountnumber)>1);

select * from ud502.u58417_h_mdl_app_mntr 
where accountnumber IN (select accountnumber from (select accountnumber, count(accountnumber) as cnt
from ud502.u58417_h_mdl_app_mntr 
group by 1)
where cnt > 1);

	
-- Find out your teradata roles permission access ;
select * from dbc.rolemembersx;

-- find out who the ID of the  owner of a  table in teradata is
select * from SYSDBA.SKEWINFO where  databasename='ud151' and tablename ='cll_addresses';

-- find the creator's name when the ID is available. There are two ways ;
-- using the dbc.databases view or the povctl.hr_itro_dt_lgin_ownr_hist_bc view ;
select * from povctl.HR_ITRO_TD_LGIN_OWNR_HIST_bc where lgin_id = '<user id>';
select * from dbc.databases where databasename = 'dzu532'; -- look in the commentstring column

-- add row number as a variable
PROC SQL;
CREATE TABLE SQL_FOR_ROW_NUM AS
SELECT *,MONOTONIC() AS ROW_NUM
FROM ROW_NUM_DATA
;
QUIT;

-- collect statistics
collect statistics ud502.mlb198_test index(app_id) ;
collect statistics ud502.mlb198_test column(src_app_id, app_datetime);

--A good skew ratio is anywhere from 0 to 1. If your skew ratio is greater than 1 find another primary index.
--Smaller tables will have naturally higher skews due to the simple fact there aren’t a lot of rows to split out.


-- find out the skewness of a table
select tablename, skewratio
from sysdba.skewinfo
where databasename = 'UD502'
and tablename = 'mlb198_test';



-- intersect columns
SELECT [COLUMN NAME 1], [COLUMN NAME 2],… FROM [TABLE NAME 1] 
INTERSECT 
SELECT [COLUMN NAME 1], [COLUMN NAME 2],… FROM [TABLE NAME 2]

-- volatile table
create volatile table dzu532_fico_gm08v1  as
(select ....
)
with data primary index (acct_id) 
;
-- volatile table with on commit instruction
create volatile table dzu532_fico_gm08v1  as
(select ....
)
with data primary index (acct_id) 
on commit preserve rows
;

-- avoid overflown error, use cast
-- select metrics for inbase accounts

  select  
            as_of_dt
           ,business_unit_cd
           ,portfolio_cd
           , cast( count(*) as decimal (18,0) )
		   , sum(  cast( total_line_amt as decimal (18,0) ) )
		   , sum(   cast( total_bal_amt as decimal (18,0) ) )
		   , sum (   cast( loc_total_bal_amt as decimal (18,0) ) )
		
    from udw.apac_monthly_hist   
    where as_of_dt = '2015-12-01'  and cm11 in (  select  cm11   from udw.apac_monthly_hist  where as_of_dt='2015-12-01'   and (total_bal_amt > 0   or aif_ind = 1 )   )  
    group by 1, 2, 3 ;      


-- join columns
SELECT column_name(s)
FROM table_name1
INNER JOIN table_name2
ON table_name1.column_name=table_name2.column_name

-- between dates in a table
SELECT a.acct, b.acct, a.stmt_date, b.open_date, a.balance, b.credit
FROM U12345_Table_A A LEFT OUTER JOIN U12345_Table_B B
ON A.acct = B.acct
WHERE A.stmt_date between '2003-01-15' and '2003-01-25';

-- alternatives to identify a sample in a table 
This is a challenging question, mainly because the best answer depends on the data/tables being used.
There are 3 primary choices:
1. Use the SAMPLE clause
2. Use the TOP operator
3. Use a driver table with the appropriate sampling of driver rows.
The TOP operator works very well for relatively small number of rows when used against a single table without any WHERE conditions, and not in a CREATE TABLE statement.  When TOP is not a good choice, the entire query is run before your data is extracted.  In these cases, a SAMPLE clause works much better.
The SAMPLE clause will cause the entire query to be run before your sample is extracted.  This can be expensive.
The best overall choice is probably to create a driver table with the appropriate number of identifiers (PI values) and using the PI column in an ACCT_ID IN type of clause (with ACCT_ID being the actual PI column).  This provides the most flexibility and is less dependent on the query, but it is dependent on the table (PI).
Thanks,
Mike
--

-- List of tables with user id, name and email of the owner
SEL databasename , tablename ,  creatorname, CreateTimeStamp, commentstring  
FROM dbc.tables
WHERE databasename = 'UD149'


-- opereators in where clause equal not equal, etc
= 	Equal
<> 	Not equal
> 	Greater than
< 	Less than
>= 	Greater than or equal
<= 	Less than or equal
BETWEEN 	Between an inclusive range
LIKE 	Search for a pattern
IN 	If you know the exact value you want to return for at least one of the columns

Note: In some versions of SQL the <> operator may be written as !=


-- union operator
select field1, field2, . field_n
from tables
UNION
select field1, field2, . field_n
from tables;

-- search for strings in column name and variable name
SQL Server search column and table name

SELECT sc.[name] AS column_name, so.[name] AS TABLE1
FROM syscolumns sc
INNER JOIN sysobjects so ON sc.id=so.id
WHERE sc.[name] LIKE 'typeid' and so.xtype = 'U'
order by 1,2

- Teradata search column and table name


SELECT * FROM dbc.COLUMNS WHERE UPPER(databasename)='PLOAN_SECURED' AND 
UPPER(columnname)='GL_DEPT_CD'

- search teradata by table name;
SELECT tablename, databasename FROM dbc.tables WHERE tablename LIKE '%TABLE/DATABASE%' ;

-- space utilization teradata ;
select
	trim(databasename) as databasename
	,sum(maxperm)/1024/1024 (decimal(10,2)) total_space
	,sum(currentperm) /1024/1024 (decimal(10,2)) used_space
	,(max(currentperm)*count(*)-sum(currentperm))/1024/1024 (decimal(10,2)) skewsize
	,(used_space + skewsize) total_used_space
	,(min(maxperm - currentperm) /1024/1024)*count(*)  (decimal(10,2)) free_space
	,(total_used_space/total_space) utilization
from
	dbc.diskspace
where
		databasename in ('UD149')  --Replace UD146 with your UD Container
	and	maxperm > 0
group by
	1
order by
	3
;
 

-- Find files that have been recently deleted: On Teradata, run the following script, replacing 'eid' with your own eid. ;
select * 
 from ud464.da_unix_files_dec21
where  login_id = 'eid'    Update here with your eid
and platform_id = 6
and  file_name not in
(select file_name from ud464.da_unix_files
where snap_dt = '2011-12-28'
and login_id = 'eid'         Update here with your eid
and platform_id = 6
group by 1)

-- Add fields taking into account missing values
select ID, (coalesce(VALUE1 ,0) + coalesce(VALUE2 ,0) as Total from TableName



-- stack multiple tables 


	 select 
				a.cm11
			, 	(case 	when a.scenario = 'A' then 'adverse'
								when a.scenario = 'B' then 'base'  
								when a.scenario = 'S' then 'severe' 
					END ) as scenario
			,	a.seg_nm as seg_nm_cap
			,	a.segment as segment_cap
			,	b.seg_nm as seg_nm_ead
			, 	b.segment as segment_ead
			,	c.seg_nm as seg_nm_el
			, 	c.segment as segment_el
			, (CASE WHEN 1=1 THEN 1 ELSE 0 END) as qtr				
			,	a.score1   as cap_idn			
			,	b.score1   as ead_idn
			,	c.score1   as el_idn

			

		from 				mse.basel_ind_uscs_st_model  a 
				left join mse.basel_ind_uscs_st_model  b
						on a.cm11 = b.cm11 and a.scenario = b.scenario 
			left join mse.basel_ind_uscs_st_model  c
					on a.cm11 = c.cm11 and a.scenario = c.scenario 
		where
			 a.seg_nm in ( 
											'USPCL00_BASELCAPEAD '
										,	'USPSC00_BASELCAPEAD'
										,	'USPSL00_BASELCAPEAD '
										,	'USPCC00_BASELCAPEAD'
										)
		and 								
					 b.seg_nm in ( 
											'USPSC00_EAD'
										,   'USPCL00_EAD'
										,   'USPSL00_EAD'
										,   'USPCC00_EAD'
										)
		and
					 c.seg_nm in ( 
											'USPSC00_ELEAD'
										,   'USPCL00_ELEAD'
										,   'USPSL00_ELEAD'
										,   'USPCC00_ELEAD'
										)								
		and a.data_run_dt = &dt2 
		and b.data_run_dt = &dt2
		and c.data_run_dt = &dt2
							


		UNION


		select 
				a.cm11
			, 	(case 	when a.scenario = 'A' then 'adverse'
								when a.scenario = 'B' then 'base'  
								when a.scenario = 'S' then 'severe' 
					END ) as scenario
			,	a.seg_nm as seg_nm_cap
			,	a.segment as segment_cap
			,	b.seg_nm as seg_nm_ead
			, 	b.segment as segment_ead
			,	c.seg_nm as seg_nm_el
			, 	c.segment as segment_el
			, (CASE WHEN 1=1 THEN 2 ELSE 0 END) as qtr				
			,	a.score2   as cap_idn			
			,	b.score2   as ead_idn
			,	c.score2   as el_idn

			

		from 				mse.basel_ind_uscs_st_model  a 
				left join mse.basel_ind_uscs_st_model  b
						on a.cm11 = b.cm11 and a.scenario = b.scenario 
			left join mse.basel_ind_uscs_st_model  c
					on a.cm11 = c.cm11 and a.scenario = c.scenario 
		where
			 a.seg_nm in ( 
											'USPCL00_BASELCAPEAD '
										,	'USPSC00_BASELCAPEAD'
										,	'USPSL00_BASELCAPEAD '
										,	'USPCC00_BASELCAPEAD'
										)
		and 								
					 b.seg_nm in ( 
											'USPSC00_EAD'
										,   'USPCL00_EAD'
										,   'USPSL00_EAD'
										,   'USPCC00_EAD'
										)
		and
					 c.seg_nm in ( 
											'USPSC00_ELEAD'
										,   'USPCL00_ELEAD'
										,   'USPSL00_ELEAD'
										,   'USPCC00_ELEAD'
										)								
		and a.data_run_dt = &dt2
		and b.data_run_dt = &dt2
		and c.data_run_dt = &dt2
							

