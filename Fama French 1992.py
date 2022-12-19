#!/usr/bin/env python
# coding: utf-8

# In[21]:


import pandas as pd
import numpy as np
import os
import copy
import datetime as dt
from dateutil.relativedelta import *
from pandas.tseries.offsets import * 


# In[22]:


#-----------------------------------
# (1) CRSP Data setup
#-----------------------------------
# read the csv and select columns
filepath = "C:/Users/Ruiqi Yuan/Desktop/PythonTest/CRSP.csv"
CRSP = pd.read_csv(filepath,usecols=['PERMNO','date','PRC','SICCD','SHROUT','EXCHCD', 'SHRCD'])
print(CRSP.head(10))
CRSP.columns = ['stock_code','date','SHRCD','exchange_code','SIC_code','price','shares']
CRSP['date'] = CRSP['date'].apply(str)
CRSP['date'] = pd.to_datetime(CRSP['date'])
CRSP['year'] = CRSP['date'].dt.year
CRSP['year_month'] = CRSP['date'].dt.to_period('M')
CRSP['price'] = abs(CRSP['price'])
print(CRSP.head(10))
#drop na
CRSP.dropna(inplace=True)
# line up date to end of month
CRSP['jdate']= CRSP['date']+MonthEnd(0)
print(CRSP.head(10))


# In[23]:


# calculate market equity(ME) ME = prc*shrout/1000 (million)
CRSP['ME'] = CRSP['price']*CRSP['shares']/1000 
print(CRSP.head(10))


# In[24]:


#Remove all obs outside July 1962 - Dec 2020
CRSP=CRSP[(CRSP['year_month']>='1962-07') & (CRSP['year_month']<='2020-12')]
print(CRSP.head(10))


# In[25]:


i = 0
r = None
for row in CRSP['year_month']:
    if str(row)<'1962-07' or str(row)>'2020-12':
        i += 1
        r = row
print(r)
print(i)


# In[26]:


#select exchanges
CRSP['exchange_code'].unique()
CRSP = CRSP.loc[CRSP['exchange_code'].isin([1,2,3]),:]
CRSP.head(10)
CRSP['exchange_code'].unique()
## keep only share code 10,11 
CRSP['SHRCD'].unique()
CRSP = CRSP.loc[CRSP['SHRCD'].isin([10.0,11.0]),:]
CRSP['SHRCD'].unique()
print(CRSP.head(10))


# In[27]:


# calculate return
CRSP['ret'] = CRSP.groupby(['stock_code'])['price'].apply(lambda x : x/x.shift() - 1)
CRSP.dropna(inplace=True)
print(CRSP.head(10))


# In[29]:


#-----------------------------------
# (2) Table 1 pre-beta, post-beta and ln(ME)
#-----------------------------------
# calculate lag size 
CRSP = CRSP.sort_values(['stock_code', 'date'])
CRSP['ME_lag'] = CRSP.groupby('stock_code')['ME'].shift(1)
# if first month of a permno, size_lag is size/(1+ret)
CRSP['ME_lag'] = np.where(CRSP['ME_lag'].isna(), CRSP['ME']/(1+CRSP['ret']), CRSP['ME_lag'])
print(CRSP.head(10))


# In[37]:


#Get 10% decile ME for each June in NYSE and asign to portfolio
CRSP['month']=CRSP['date'].dt.month
q1 = CRSP[(CRSP['month']==3)|(CRSP['month']==6)|(CRSP['month']==9)|(CRSP['month']==12)]
q1 = q1[['date', 'jdate','stock_code', 'exchange_code', 'ME']]

# all monthly observations
m1 = CRSP[['date','jdate','stock_code','ME_lag', 'ret']]
m1['port_date']=m1['jdate']+QuarterEnd(-1)

# quarterly NYSE observations
nyse_q = q1[q1['exchange_code']==1]

# Calculate NYSE Decile Break Point
nyse_bp=nyse_q.groupby(['jdate'])['ME'].describe(percentiles=[0, .1, .2, .3, .4,.5,.6,.7,.8,.9, 1]).reset_index()

nyse_bp=nyse_bp.rename(columns={'0%':'pct0', '10%':'pct10', '20%':'pct20','30%':'pct30',                                 '40%':'pct40','50%':'pct50','60%':'pct60','70%':'pct70',                                '80%':'pct80','90%':'pct90','100%':'pct100'})

nyse_bp = nyse_bp.drop(['count','mean','std','min','max'], axis=1)
nyse_bp['qtrdate'] = nyse_bp['jdate']+QuarterEnd(0)


# In[36]:


# Build portfolios based on ME 
# assign stocks to portfolios at each quarter end
qtr_decile = pd.merge(q1, nyse_bp, how='left', left_on=['jdate'], right_on=['qtrdate'])
qtr_decile = qtr_decile.drop(['jdate_x','jdate_y'], axis=1)

# function to assign size group
def sizegrp(row):
    if   (row['pct0']<=row['ME']<row['pct10']):
        group=1
    elif (row['pct10']<=row['ME']<row['pct20']):
        group=2
    elif (row['pct20']<=row['ME']<row['pct30']):
        group=3
    elif (row['pct30']<=row['ME']<row['pct40']):
        group=4
    elif (row['pct40']<=row['ME']<row['pct50']):
        group=5
    elif (row['pct50']<=row['ME']<row['pct60']):
        group=6
    elif (row['pct60']<=row['ME']<row['pct70']):
        group=7
    elif (row['pct70']<=row['ME']<row['pct80']):
        group=8
    elif (row['pct80']<=row['ME']<row['pct90']):
        group=9
    elif (row['pct90']<=row['ME']<=row['pct100']):
        group=10
    else:
        group=np.nan
    return pd.Series({'stock_code':row['stock_code'], 'date':row['date'], 'exchange_code':row['exchange_code'],                       'qtrdate':row['qtrdate'], 'ME': row['ME'], 'group': group})
        
qtr_decile=qtr_decile.apply(sizegrp, axis=1)

# label NYSE listed stocks
qtr_decile['nyse_listed']=np.where(qtr_decile['exchange_code']==1, 'yes', 'no')
# summary table for frequency count
summ_table = qtr_decile.groupby(['qtrdate', 'group','nyse_listed'])['stock_code'].count().reset_index()
freq_table=pd.pivot_table(summ_table, index=['qtrdate'], values='stock_code', columns=['nyse_listed','group'])
freq_table.head(10)


# In[39]:


###########################################
# Establish Size Group for Monthly Record #
###########################################

m_group = pd.merge(m1[['stock_code','date','ret','ME_lag','port_date']],                    qtr_decile[['stock_code','qtrdate','group']],                    how='left', left_on=['stock_code', 'port_date'], right_on=['stock_code','qtrdate'])
m_group=m_group[m_group['group'].notna()].drop(['qtrdate','port_date'], axis=1)
# Calculate Monthly Weighted Average Returns by Size Group
m_group =m_group.sort_values(['group', 'date'])

# function to calculate value weighted return
def wavg(group, avg_name, weight_name):
    d = group[avg_name]
    w = group[weight_name]
    try:
        return (d * w).sum() / w.sum()
    except ZeroDivisionError:
        return np.nan

# value-weigthed return
dec_ret=m_group.groupby(['group','date']).apply(wavg, 'ret','ME_lag').to_frame().reset_index().rename(columns={0: 'decile_ret'})


# In[40]:


# function to get coefficients (calculate pre-ranking beta，rolling windows = [24,60]) 
import statsmodels.api as sm

class rollingRegression():
    '''This is a class for rolling OLS'''
    def __init__(self,y,x,windowlength,group):
        self.y= y
        self.x = x
        


# In[46]:


#-----------------------------------
# (3) COMPUSTAT Data setup and combining
#-----------------------------------
filepath = r"C:/Users/Ruiqi Yuan/Desktop/PythonTest/COMPUSTAT.csv"
value = pd.read_csv(filepath,usecols=['gvkey','datadate','fyear','bkvlps','ceq','csho','dvp','ib','txdb','txdfed','txdfo','txds','prcc_f'])
value['date'] = value['datadate'].apply(str)
value['date'] = pd.to_datetime(value['date'])
value['year'] = value['date'].dt.year
value.head()
value.dropna(inplace=True)
value.head()




# In[49]:



#-----------------------------------
# (4) Calculation
#-----------------------------------
# Calculate Market Equity (ME) : ME = prc*shrout
shares = copy.deepcopy(CRSP.loc[:,['stock_code','exchange_code',
                                    'price','shares','year','year_month']])
shares.head()

shares = shares.sort_values(['stock_code','year_month'])
shares = shares.groupby(['stock_code','year']).last()
shares.reset_index(inplace=True)
shares.dropna(inplace=True)
shares.head()

# Calculate Book Value of Common Equity (BE) : BE = book value of common equity + deferred taxes (t-1 December data)BE = CEQ + TXDB lag 1 but group by gvkey?
value['BE'] =value['ceq'] + value['txdb'] 


# Calculate Total Book Assets (A) : A = total book assets (t-1 December data)



# Calculate Earnings (E) : E = income before extraordinary items + income-statement deferred taxes - preferred dividends (t-1 December data)EP = IB + TXDFED + TXDFO + TXDS -  DVP/PRCC_F
value['E'] = value['ib'] + value['txdfed'] + value['txdfo'] + value['txds'] - value['dvp'] /value['prcc_f'] 
value.head()


# Firm size ln(ME) (t June data)


# In[ ]:




