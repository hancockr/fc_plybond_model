# Databricks notebook source
pip install imblearn

# COMMAND ----------

pip install xgboost

# COMMAND ----------

pip install lime

# COMMAND ----------

pip install shap

# COMMAND ----------

import pyspark
from pyspark.sql.functions import to_date, unix_timestamp, from_unixtime, mean, col, stddev, concat, lit, min, isnull, isnan, when, sum, lag, weekofyear, max, exp, count, log, date_sub, next_day, count, substring, countDistinct, upper, lag
from pyspark.sql.window import Window
from pyspark.sql.types import IntegerType, FloatType, DateType
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import math
import datetime
import scipy.stats
import warnings
import imblearn
from sklearn import linear_model
from sklearn import metrics
from sklearn.linear_model import Ridge, Lasso, MultiTaskLasso
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ParameterGrid
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeRegressor
from sklearn import tree
from sklearn.ensemble import BaggingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import EllipticEnvelope
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectFromModel
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from datetime import timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR,SVC
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.tree import plot_tree, export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler, TomekLinks, NearMiss
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.tree import _tree, export_text
from xgboost import XGBClassifier
import lime
from lime import lime_tabular
import shap
from matplotlib.ticker import PercentFormatter

# COMMAND ----------

# DBTITLE 1,Variable/Tag ID Lists
# This block of code contains all the dictionaries associated with the variable tags for MK81 and the paper machines.

# MK 81
variableListMK81_cl = {
    31721 : 'PVA Flow',
    66061 : 'Online Glue Temp',
    31714 : 'Line Speed',
    177326 : 'App. Gap DS',
    177327 : 'App. Gap OS',
    177325 : 'Recal. Gap OS',
    177324 : 'Recal. Gap DS',
    31717 : 'Rt Load Cell',
    31718 : 'Lt Load Cell',
    46664 : 'UWS Normal Tension',
    46381 : 'Winder Tension'
}
variableListMK81_qa = {
    31524 : 'PRID',
    161960 : 'Plybond Test Average',
    161961: 'Sheet 01',
    161962: 'Sheet 02',
    161963: 'Sheet 03',
    161964: 'Sheet 04',
    194557: 'RP2 Plybond Average',
    194544: 'RP2 Sheet 01',
    194545: 'RP2 Sheet 02',
    194546: 'RP2 Sheet 03',
    194547: 'RP2 Sheet 04',
    194530: 'RP5 Plybond Average',
    194548: 'RP5 Sheet 01',
    194549: 'RP5 Sheet 02',
    194550: 'RP5 Sheet 03',
    194551: 'RP5 Sheet 04',
    194558: 'RP8 Plybond Average',
    194552: 'RP8 Sheet 01',
    194553: 'RP8 Sheet 02',
    194554: 'RP8 Sheet 03',
    194555: 'RP8 Sheet 04',
    185950: 'Average Emboss'
}
# Data side loaded for MK81

variableListMK81_side = {
    175664: 'SER/App. Gap OS',
    175665: 'SER/App. Gap Center',
    175666: 'SER/App. Gap DS',
    175679: 'SER/App Gap Average',
    175667: 'Marrying Nip OS',
    175668: 'Marrying Nip Center',
    175669: 'Marrying Nip DS',
    175677: 'Marrying Nip Avg',
    175670: 'Pressure Nip OS',
    175671: 'Pressure Nip Center',
    175672: 'Pressure Nip DS',
    175678: 'Pressure Nip Avg',
    32000: 'Anilox Nip OS',
    32001: 'Anilox Nip Center',
    32002: 'Anilox Nip DS',
    57290: 'Glue % Solids',
    112387: 'Cleaned PVA Glue Reservoir'
}

# 1M
variableListMP1_cl ={
    42845 : 'Reel Moisture',
#    '167947' : '% Unrefined' ,
    167946 : '% Refined',
    168273 : '% Broke',
    43293 : '% SSK',
    168270 : '% NSK',
    168272 : '% Euc',
    42817 : 'Wire to Press Draw',
    170598 : 'Yankee to Reel Draw',
    42785 : 'Headbox Pressure',
    42881 : 'PUS Vacuum',
    43804 : 'PRID',
    43216 : 'Absorb Aid',
    43311 : 'CMC',
    43571 : 'Kymene',
    43359 : 'Emulsion',
    42914 : 'Yankee Speed',
    42910 : 'Reel Speed',
    44576 : 'Creping Blade Life',
    44570 : 'Cleaning Blade Life'
}
variableListMP1_qa = {
    44370 : 'Stretch',
    44391 : 'Tensiles',
    44395 : 'Wet Burst',
    44264 : 'Basis Weight',
    59355 : 'Caliper A Roll',
    59356 : 'Caliper B Roll',
    163652 : 'Caliper Range',
    44379 : 'Tensile Modulus CD',
    44388 : 'Tensile Ratio',
    72753 : 'Work to Tear',
    66470 : 'Water Alkalinity'
}

variableListMP1_side = {
    43804: 'PRID',
    44570: 'Cleaning Blade Life'
}

# 2M
variableListMP2_cl = {
    25373 : 'Reel Moisture',
#    '167986' :'% Unrefined' ,
    167985 : '% Refined',
    26028 : '% Broke',
    25769 : '% SSK',
    25947 : '% NSK',
    26053 : '% Euc',
    25323 : 'Wire to Press Draw',
    25324 : 'Yankee to Reel Draw',
    25289 : 'Headbox Pressure',
    25414 : 'PUS Vacuum',
    27304 : 'PRID',
    25661 : 'Absorb Aid',
    25819 : 'CMC',
    26165 : 'Kymene',
    25851 : 'Emulsion',
    25360 : 'Yankee Speed',
    27415 : 'Reel Speed',
    25464 : 'Creping Blade Life',
    25444 : 'Cleaning Blade Life',
    66470 : 'Water Alkalinity'
 }
variableListMP2_qa ={
    27064 : 'Stretch',
    27085 : 'Tensiles',
    27089 : 'Wet Burst',
    25370 : 'Basis Weight',
    59379 : 'Caliper A Roll',
    59380 : 'Caliper B Roll',
    26579 : 'Caliper Range',
    27073 : 'Tensile Modulus CD',
    27082 : 'Tensile Ratio',
    72800 : 'Work to Tear'
}

# 3M
variableListMP3_cl = {
    170709 : 'Reel Moisture',
#    '168003' :'% Unrefined' ,
    168002 : '% Refined',
    10437 : '% Broke',
    10221 : '% SSK',
    10385 : '% NSK',
    10456 : '% Euc',
    9871 : 'Wire to Press Draw',
    9900 : 'Yankee to Reel Draw',
    9948 : 'PUS Vacuum',
    10744 : 'PRID',
    10136 : 'Absorb Aid',
    10251 : 'CMC',
    205884 : 'Kymene',
    10298 : 'Emulsion',
    9898 : 'Yankee Speed',
    9895 : 'Reel Speed',
    9987 : 'Creping Blade Life',
    9968 : 'Cleaning Blade Life',
    66470 : 'Water Alkalinity'
 }
variableListMP3_qa ={
    11531 : 'Stretch',
    11535 : 'Tensiles',
    11540 : 'Wet Burst',
    10782 : 'Basis Weight',
    59403 : 'Caliper A Roll',
    59404 : 'Caliper B Roll',
    11045 : 'Caliper Range',
    11519 : 'Tensile Modulus CD',
    11531 : 'Tensile Ratio',
    177112 : 'Work to Tear'
}
variableListMK3_hist = {
    'MK2MIH.CM_2HDBXPR_DACA_PV.F_CV': 'Headbox Pressure'
}
# 4M
variableListMP4_cl = {
    170734 : 'Reel Moisture',
#    '167952' :'% Unrefined' ,
    167951 : '% Refined',
    32845 : '% Broke',
    167962 : '% SSK',
    167963 : '% NSK',
    167965 : '% Euc',
    32266 : 'Wire to Press Draw',
    32268 : 'Yankee to Reel Draw',
    32220 : 'Headbox Pressure',
    32356 : 'PUS Vacuum',
    33179 : 'PRID',
    32539 : 'Absorb Aid',
    32651 : 'CMC',
    32928 : 'Kymene',
    32687 : 'Emulsion',
    32314 : 'Yankee Speed',
    32305 : 'Reel Speed',
    32409 : 'Creping Blade Life',
    32389 : 'Cleaning Blade Life',
    66470 : 'Water Alkalinity'
 }
variableListMP4_qa ={
    33851 : 'Stretch',
    33857 : 'Tensiles',
    33876 : 'Wet Burst',
    33788 : 'Basis Weight',
    59427 : 'Caliper A Roll',
    59428 : 'Caliper B Roll',
    33432 : 'Caliper Range',
    33860 : 'Tensile Modulus CD',
    33869 : 'Tensile Ratio',
    72938 : 'Work to Tear'
}

# 7M
variableListMP7_cl = {
    170762 : 'Reel Moisture',
#    '22374' :'% Unrefined' ,
    65534 : '% Refined',
    22462 : '% Broke',
    173727 : '% SSK',
    173726 : '% NSK',
    173729 : '% Euc',
    170759 : 'Wire to Press Draw',
    170761 : 'Yankee to Reel Draw',
    21578 : 'Headbox Pressure',
    21914 : 'PUS Vacuum',
    22793 : 'PRID',
    22116 : 'Absorb Aid',
    22221 : 'CMC',
    22509 : 'Kymene',
    22237 : 'Emulsion',
    21754 : 'Yankee Speed',
    21739 : 'Reel Speed',
    22023 : 'Creping Blade Life',
    22003 : 'Cleaning Blade Life',
    66470 : 'Water Alkalinity'
 }
variableListMP7_qa ={
    23512 : 'Stretch',
    23540 : 'Tensiles',
    23545 : 'Wet Burst',
    23400 : 'Basis Weight',
    59499 : 'Caliper A Roll',
    59500 : 'Caliper B Roll',
    62443 : 'Caliper Range',
    23767 : 'Tensile Modulus CD',
    23536 : 'Tensile Ratio',
    73083 : 'Work to Tear'
}

# COMMAND ----------

# DBTITLE 1,Obtaining Raw Data and Analysis Associated


# COMMAND ----------

# I created a time frame. I was having difficulty making sure that all of the 15 minute blocks were accounted for when joining. This just allowed me to ensure that the 15 minute bucket were all consistent so I was not skipping a bucket when filling.

timeframe = spark.sql("SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes))")
timeframe = timeframe.toPandas()
timeframe.columns = ['TimeBucket']
timeframe['TimeBucket'] = pd.to_datetime(timeframe['TimeBucket'])
timeframe = timeframe.sort_values(by = 'TimeBucket')
timeframe['Hold'] = 1
timeframe = timeframe.set_index('TimeBucket')

# COMMAND ----------

# Side loaded variables: all of the sideloaded variables until 7/18/23
sparkdf = spark.sql("select * from groupdb_famc_qa.plybond_additional_vars")
side = sparkdf.toPandas()

# COMMAND ----------

# DBTITLE 1,MK81
# This gathers all of the variables from MK81's different tables. The parent roll id is gathered from a different query because when averaging, I would loose the PRID

# mk81 centerline variables
sparkdfmk81cl = spark.sql("select tb.TimeBucket, cl.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MK81' AND (cl.var_id IN ('31721','66061' , '31714' ,'177326' ,'177327' ,'177325' ,'177324' ,'31717' ,'31718' ,'46664','46381')) GROUP BY cl.var_id , tb.TimeBucket")
mk81_cl = sparkdfmk81cl.toPandas()
# mk81 quality variables
sparkdfmk81qa = spark.sql("select tb.TimeBucket, qa.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.qatasks_data AS qa on qa.result_on >= tb.TimeBucket AND (qa.result_on < (tb.TimeBucket + interval 15 minutes)) AND qa.pl_desc = 'TT MK81' AND (qa.var_id IN ('161960' ,'161961','161962','161963','161964','194557','194544','194545','194546','194547','194530','194548','194549','194550','194551','194558','194552','194553','194554','194555','185950')) GROUP BY qa.var_id , tb.TimeBucket")
mk81_qa = sparkdfmk81qa.toPandas()
# parent roll id for MK81
sparkdfmk81prid = spark.sql("select tb.TimeBucket, result FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.qatasks_data AS qa on qa.result_on >= tb.TimeBucket AND (qa.result_on < (tb.TimeBucket + interval 15 minutes)) AND qa.pl_desc = 'TT MK81' AND (qa.var_id IN ('31524'))")
mk81_prid = sparkdfmk81prid.toPandas()
# mk81
mk81_cl1 = mk81_cl
mk81_qa1 = mk81_qa
mk81_prid1 = mk81_prid

# COMMAND ----------

# This block of code gathers the side loaded variables from MK81 and does data manipulation to get the desired result. The cleaning variable is a PASS/FAIL variable, so I replaced a PASS/FAIL with the time that that was recorded. Then forward filled to get the amount of time since the equipment was cleaned in that time bucket. 

mk81_side = side.loc[side['pl_desc'] == 'TT MK81']
mk81_extraprid = mk81_side.loc[mk81_side['var_id'] == 31524]
mk81_clean = mk81_side.loc[mk81_side['var_id'] == 112387]
mk81_side = mk81_side[mk81_side.var_id != 31524]
mk81_side = mk81_side[mk81_side.var_id != 112387]
mk81_side['result'] = mk81_side['result'].astype('float')
mk81_side = mk81_side.pivot_table( values = ['result'], columns = ['var_id'], index = ['result_on'])
mk81_side = mk81_side.reset_index()
mk81_side.columns = mk81_side.columns.droplevel(0)
mk81_side.columns.name = None
mk81_side.columns = ['TimeBucket',32000,32001,32002,57290,175664,175665,175666,175667,175668,175669,175670,175671,175672,175677,175678,175679]
mk81_side = mk81_side.rename(columns = variableListMK81_side)
mk81_side['TimeBucket'] = pd.to_datetime(mk81_side['TimeBucket'])
mk81_side['TimeBucket'] = mk81_side['TimeBucket'].dt.round('15min')
mk81_side = mk81_side.sort_values(by = 'TimeBucket')
mk81_side = mk81_side.groupby('TimeBucket').mean()
mk81_clean['result'] = [1 if x == 'Pass' else 0 for x in mk81_clean['result']]
mk81_clean = mk81_clean.loc[mk81_clean['result'] == 1]
mk81_clean['result_on'] = pd.to_datetime(mk81_clean['result_on'])
mk81_clean = mk81_clean.drop_duplicates()
mk81_clean = mk81_clean.pivot(index = 'result_on',columns = 'var_id',values = 'result')
mk81_clean = mk81_clean.reset_index()
mk81_clean.columns = ['TimeBucket','Cleaned']
mk81_clean['Cleaned'] = mk81_clean['TimeBucket']
mk81_clean['TimeBucket'] = mk81_clean['TimeBucket'].dt.round('15 min')
mk81_clean = mk81_clean.set_index('TimeBucket')
mk81_clean = timeframe.join(mk81_clean, how = 'outer')
mk81_clean['Cleaned'] = mk81_clean['Cleaned'].ffill()
mk81_clean = mk81_clean.reset_index()
mk81_clean['Cleaned'] = (mk81_clean['TimeBucket'] - mk81_clean['Cleaned'])/pd.Timedelta(minutes = 1)
mk81_clean = mk81_clean.drop(columns = 'Hold')
mk81_clean = mk81_clean.sort_values(by = 'TimeBucket')
mk81_clean = mk81_clean.set_index('TimeBucket')
mk81_clean = mk81_clean.dropna()
mk81_clean['Cleaned'] = [0 if x < 0 else x for x in mk81_clean['Cleaned']]

# COMMAND ----------

# Gathering all of the variables from the centerline table
mk81_cl = mk81_cl1.pivot(index = 'TimeBucket', columns = 'var_id', values = 'avg(result)')
mk81_cl = mk81_cl.rename(columns = variableListMK81_cl)
mk81_cl = mk81_cl.reset_index()
mk81_cl['TimeBucket'] = pd.to_datetime(mk81_cl['TimeBucket'])
mk81_cl = mk81_cl.sort_values(by = 'TimeBucket')
mk81_cl = mk81_cl.set_index('TimeBucket')

# COMMAND ----------

# Plybond test values: need to combine the RP2/5/8 into an average so consistent before 4/8

mk81_qa = mk81_qa1.pivot(index = 'TimeBucket', columns = 'var_id', values = 'avg(result)')
mk81_qa = mk81_qa.rename(columns = variableListMK81_qa)
mk81_qa = mk81_qa.reset_index()
mk81_qa['TimeBucket'] = pd.to_datetime(mk81_qa['TimeBucket'])
mk81_qa = mk81_qa.sort_values(by = 'TimeBucket')
mk81_qa = mk81_qa.set_index('TimeBucket')
ply_before = mk81_qa[['Plybond Test Average','Sheet 01','Sheet 02','Sheet 03','Sheet 04']]
ply_after = mk81_qa[['RP2 Plybond Average','RP5 Plybond Average','RP8 Plybond Average']]
ply_before = ply_before.dropna()
ply_after = ply_after.dropna()
ply_before['Standard Deviation'] = ply_before[['Sheet 01','Sheet 02','Sheet 03','Sheet 04']].std(axis = 1)
#ply_after['Plybond Test Average'] = ply_after.mean(axis = 1)
ply_after['Plybond Test Average'] = ply_after.min(axis = 1)
ply_after['Standard Deviation'] = ply_after[['RP2 Plybond Average','RP5 Plybond Average','RP8 Plybond Average']].std(axis = 1)
ply_before = ply_before.drop(columns = ['Sheet 01','Sheet 02','Sheet 03','Sheet 04'])
ply_after = ply_after.drop(columns = ['RP2 Plybond Average','RP5 Plybond Average','RP8 Plybond Average'])
ply = pd.concat([ply_before,ply_after])
mk81_qa = mk81_qa.drop(columns = 'Plybond Test Average')
mk81_qa = mk81_qa.join(ply)
mk81_qa = mk81_qa.drop(columns = ['Sheet 01','Sheet 02','Sheet 03','Sheet 04','RP2 Plybond Average','RP5 Plybond Average','RP8 Plybond Average'])

# COMMAND ----------

p = ply.reset_index()
py = p[(p['TimeBucket'] > '2023-01-01') ]
len(py)

# COMMAND ----------

6/663

# COMMAND ----------

12/1140

# COMMAND ----------

# Gathering and cleaning the PRIDs

mk81_prid = mk81_prid1.replace('No Running PR', np.nan)
mk81_prid['TimeBucket'] = pd.to_datetime(mk81_prid['TimeBucket'])
mk81_prid = mk81_prid.sort_values(by = 'TimeBucket')
mk81_prid = mk81_prid.set_index('TimeBucket')
mk81_prid.columns = ['PRID']
#mk81_prid = mk81_prid.drop_duplicates()
mk81_prid = mk81_prid.reset_index()
mk81_prid = mk81_prid.drop_duplicates(subset = ['TimeBucket'], keep = 'last')
mk81_prid = mk81_prid.set_index('TimeBucket')

# COMMAND ----------

# Joining all variables into MK81

mk81 = timeframe.join(mk81_qa, how = 'outer')
mk81 = mk81.join(mk81_cl, how = 'outer')
mk81 = mk81.join(mk81_clean, how = 'outer')
mk81 = mk81.join(mk81_side, how = 'outer')
mk81 = mk81.join(mk81_prid, how = 'outer')
mk81 = mk81.drop(columns = ['Hold'])

# COMMAND ----------

# Forward filling before dropping rows without the plybond. This is because the centerline/quality data does not always line up with the timebucket the plybond test is in so we foward fill before we drop

mk81[mk81.columns.difference(['Plybond Test Average','Standard Deviation'])] = mk81[mk81.columns.difference(['Plybond Test Average','Standard Deviation'])].ffill()
mk81['Average Emboss'] = mk81['Average Emboss'].ffill()
mk81.dropna(subset = ['Plybond Test Average'])

# COMMAND ----------

mk81.columns

# COMMAND ----------

# DBTITLE 1,1M
# 1M centerline variables
sparkdf1mcl = spark.sql("select tb.TimeBucket, cl.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP1M' AND (cl.var_id IN ('42845','167946','168273','43293','168270','168272','42817','170598','42785','42881','43804','43216','43311','43571','43359','42914','42910','44576','44570')) GROUP BY cl.var_id , tb.TimeBucket")
m1_cl = sparkdf1mcl.toPandas()
# 1M quality variables
sparkdf1mqa = spark.sql("select tb.TimeBucket, qa.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.qatasks_data AS qa on qa.result_on >= tb.TimeBucket AND (qa.result_on < (tb.TimeBucket + interval 15 minutes)) AND qa.pl_desc = 'TT MP1M' AND (qa.var_id IN ('44370','44391','44395','44264','59355','59356','163652','44379','44388','72753','66470')) GROUP BY qa.var_id , tb.TimeBucket")
m1_qa = sparkdf1mqa.toPandas()
# parent roll id for 1M
sparkdf1mprid = spark.sql("select tb.TimeBucket, result, prod_code FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP1M' AND (cl.var_id IN ('43804'))")
m1_prid = sparkdf1mprid.toPandas()
# historian for 1M
# 1m sideloaded variables
m1_cl1 = m1_cl
m1_qa1 = m1_qa
m1_prid1 = m1_prid
m1_side = side.loc[side['pl_desc'] == 'TT MP1M']
m1_extraprid = m1_side.loc[m1_side['var_id'] == 43804]

# COMMAND ----------

m1_side = m1_side[m1_side.var_id != 43804]
m1_side['result'] = m1_side['result'].astype('float')
m1_side['result_on'] = pd.to_datetime(m1_side['result_on'])
m1_side = m1_side.rename(columns = {'result_on': 'TimeBucket'})
m1_side = m1_side.pivot_table(index = 'TimeBucket', columns = 'var_id', values = 'result')
m1_side = m1_side.reset_index()
m1_side.columns.name = None
m1_side.columns = ['TimeBucket','Creper Life']
m1_side['Creper Life'] = m1_side['TimeBucket']
m1_side['TimeBucket'] = m1_side['TimeBucket'].dt.round('15min')
m1_side = m1_side.groupby('TimeBucket').mean()
m1_side = timeframe.join(m1_side, how = 'outer')
m1_side = m1_side.drop(columns = ['Hold'])
m1_side['Creper Life'] = m1_side['Creper Life'].ffill()
m1_side = m1_side.reset_index()
m1_side['Creper Life'] = (m1_side['TimeBucket'] - m1_side['Creper Life'])/pd.Timedelta(minutes = 1)
m1_side['Creper Life'] = [0 if x < 0 else x for x in m1_side['Creper Life']]
m1_side = m1_side.set_index('TimeBucket')

# COMMAND ----------

m1_extraprid['TimeBucket'] = pd.to_datetime(m1_extraprid['result_on'])
m1_extraprid = m1_extraprid.sort_values(by = 'TimeBucket')
m1_extraprid['TimeBucket'] = m1_extraprid['TimeBucket'].dt.round('15min')
m1_extraprid = m1_extraprid.drop(columns = ['result_on','var_id','pl_desc'])
m1_extraprid.columns = ['PRID','TimeBucket']

# COMMAND ----------

m1_cl = m1_cl1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m1_cl = m1_cl.rename(columns = variableListMP1_cl)
m1_cl = m1_cl.reset_index()
m1_cl['TimeBucket'] = pd.to_datetime(m1_cl['TimeBucket'])
m1_cl = m1_cl.sort_values(by = 'TimeBucket')
m1_cl_clean = m1_cl[['TimeBucket','Cleaning Blade Life']]
m1_cl_clean = m1_cl_clean.dropna(subset = ['Cleaning Blade Life'])
m1_cl_clean['Cleaning Blade Life'] = m1_cl_clean['TimeBucket']
m1_cl_clean = m1_cl_clean.set_index('TimeBucket')
m1_cl = m1_cl.drop(columns = ['Cleaning Blade Life'])
m1_cl = m1_cl.set_index('TimeBucket')
m1_cl = m1_cl.join(m1_cl_clean, how = 'outer')
m1_qa = m1_qa1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m1_qa = m1_qa.rename(columns = variableListMP1_qa)
m1_qa = m1_qa.reset_index()
m1_qa['TimeBucket'] = pd.to_datetime(m1_qa['TimeBucket'])
m1_qa = m1_qa.sort_values(by = 'TimeBucket')
m1_qa = m1_qa.set_index('TimeBucket')

# COMMAND ----------

m1_prid['TimeBucket'] = pd.to_datetime(m1_prid['TimeBucket'])
m1_prid = m1_prid.sort_values(by = 'TimeBucket')
m1_prid.columns = ['TimeBucket','PRID','Product Code']

# COMMAND ----------

m1_prid2 = m1_prid
m1_prid2 = m1_prid2.drop(columns = ['Product Code'])
m1_prid2 = pd.concat([m1_extraprid,m1_prid2])

# COMMAND ----------

mp1m = timeframe.join(m1_cl)
mp1m = mp1m.join(m1_qa)
mp1m = mp1m.drop(columns = ['Hold'])
mp1m = mp1m.ffill()
mp1m = mp1m.drop(columns = ['Work to Tear', 'Tensile Modulus CD'])
mp1m = mp1m.join(m1_side, how = 'outer')
mp1m.columns

# COMMAND ----------

mp1m2 = mp1m.reset_index()
mp1m2['Cleaning Blade Life'] = mp1m2['TimeBucket'] - mp1m2['Cleaning Blade Life']
mp1m2['Cleaning Blade Life'] = mp1m2['Cleaning Blade Life']/pd.Timedelta(minutes = 1)

# COMMAND ----------

# DBTITLE 1,2M
# 2M centerline variables
sparkdf2mcl = spark.sql("select tb.TimeBucket, cl.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP2M' AND (cl.var_id IN ('25373' ,'167985' ,'26028' ,'25769' ,'25947' ,'26053' ,'25323' ,'25324' ,'25289' ,'25414'  ,'25661' ,'25819' ,'26165' ,'25851' ,'25360' ,'27415' ,'25464','25444' ,'66470')) GROUP BY cl.var_id , tb.TimeBucket")
m2_cl = sparkdf2mcl.toPandas()
# 2M quality variables
sparkdf2mqa = spark.sql("select tb.TimeBucket, qa.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.qatasks_data AS qa on qa.result_on >= tb.TimeBucket AND (qa.result_on < (tb.TimeBucket + interval 15 minutes)) AND qa.pl_desc = 'TT MP2M' AND (qa.var_id IN ('27064','27085','27089','25370','59379','59380','26579', '27073','27082','72800')) GROUP BY qa.var_id , tb.TimeBucket")
m2_qa = sparkdf2mqa.toPandas()
# parent roll id for 2M
sparkdf2mprid = spark.sql("select tb.TimeBucket, result, prod_code FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP2M' AND (cl.var_id IN ('27304'))")
m2_prid = sparkdf2mprid.toPandas()
# historian for 2M
# 2m sideloaded variables
m2_cl1 = m2_cl
m2_qa1 = m2_qa
m2_prid1 = m2_prid
m2_side = side.loc[side['pl_desc'] == 'TT MP2M']
m2_extraprid = m2_side.loc[m2_side['var_id'] == 27304]

# COMMAND ----------

m2_side = side.loc[side['pl_desc'] == 'TT MP2M']
m2_extraprid = m2_side.loc[m2_side['var_id'] == 27304]

# COMMAND ----------

m2_side = m2_side[m2_side.var_id != 27304]
m2_side['result'] = m2_side['result'].astype('float')
m2_side['result_on'] = pd.to_datetime(m2_side['result_on'])
m2_side = m2_side.rename(columns = {'result_on': 'TimeBucket'})
m2_side = m2_side.pivot_table(index = 'TimeBucket', columns = 'var_id', values = 'result')
m2_side = m2_side.reset_index()
m2_side.columns.name = None
m2_side.columns = ['TimeBucket','Creper Life', 'Caliper A Roll','Caliper B Roll']
m2_side['Creper Life'] = m2_side['TimeBucket']
m2_side['TimeBucket'] = m2_side['TimeBucket'].dt.round('15min')
m2_side = m2_side.set_index('TimeBucket')
m2_side = timeframe.join(m2_side, how = 'outer')
m2_side = m2_side.drop(columns = ['Hold'])
m2_side = m2_side.ffill()
m2_side = m2_side.reset_index()
m2_side['Creper Life'] = (m2_side['TimeBucket'] - m2_side['Creper Life'])/pd.Timedelta(minutes = 1)
m2_side['Creper Life'] = [0 if x < 0 else x for x in m2_side['Creper Life']]
m2_side = m2_side.set_index('TimeBucket')

# COMMAND ----------

m2_extraprid['TimeBucket'] = pd.to_datetime(m2_extraprid['result_on'])
m2_extraprid = m2_extraprid.sort_values(by = 'TimeBucket')
m2_extraprid['TimeBucket'] = m2_extraprid['TimeBucket'].dt.round('15min')
m2_extraprid = m2_extraprid.drop(columns = ['result_on','var_id','pl_desc'])
m2_extraprid.columns = ['PRID','TimeBucket']

# COMMAND ----------

m2_cl = m2_cl1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m2_cl = m2_cl.rename(columns = variableListMP2_cl)
m2_cl = m2_cl.reset_index()
m2_cl['TimeBucket'] = pd.to_datetime(m2_cl['TimeBucket'])
m2_cl = m2_cl.sort_values(by = 'TimeBucket')
m2_cl_clean = m2_cl[['TimeBucket','Cleaning Blade Life']]
m2_cl_clean = m2_cl_clean.dropna(subset = ['Cleaning Blade Life'])
m2_cl_clean['Cleaning Blade Life'] = m2_cl_clean['TimeBucket']
m2_cl_clean = m2_cl_clean.set_index('TimeBucket')
m2_cl = m2_cl.drop(columns = ['Cleaning Blade Life'])
m2_cl = m2_cl.set_index('TimeBucket')
m2_cl = m2_cl.join(m2_cl_clean, how = 'outer')
m2_qa = m2_qa1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m2_qa = m2_qa.rename(columns = variableListMP2_qa)
m2_qa = m2_qa.reset_index()
m2_qa['TimeBucket'] = pd.to_datetime(m2_qa['TimeBucket'])
m2_qa = m2_qa.sort_values(by = 'TimeBucket')
m2_qa = m2_qa.set_index('TimeBucket')
m2_prid['TimeBucket'] = pd.to_datetime(m2_prid['TimeBucket'])
m2_prid = m2_prid.sort_values(by = 'TimeBucket')
m2_prid.columns = ['TimeBucket','PRID','Product Code']
m2_prid2 = m2_prid
m2_prid2 = m2_prid2.drop(columns = ['Product Code'])
m2_prid2 = pd.concat([m2_extraprid,m2_prid2])
mp2m = timeframe.join(m2_cl)
mp2m = mp2m.join(m2_qa)
mp2m = mp2m.drop(columns = ['Hold'])
mp2m = mp2m.ffill()
mp2m = mp2m.drop(columns = ['Work to Tear', 'Tensile Modulus CD'])
m2_side = m2_side.drop(columns = ['Caliper A Roll','Caliper B Roll'])
mp2m = mp2m.join(m2_side, how = 'outer')
mp2m2 = mp2m.reset_index()
mp2m2['Cleaning Blade Life'] = mp2m2['TimeBucket'] - mp2m2['Cleaning Blade Life']
mp2m2['Cleaning Blade Life'] = mp2m2['Cleaning Blade Life']/pd.Timedelta(minutes = 1)

# COMMAND ----------

# DBTITLE 1,3M
# 3M centerline variables
sparkdf3mcl = spark.sql("select tb.TimeBucket, cl.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP3M' AND (cl.var_id IN (170709, 168002 ,    10437 ,    10221,    10385 ,    10456,    9871 ,    9900 ,    9948 ,    10136 ,    10251,    205884 ,    10298 ,    9898 ,    9895 ,    9968 )) GROUP BY cl.var_id , tb.TimeBucket")
m3_cl = sparkdf3mcl.toPandas()
# 3M quality variables
sparkdf3mqa = spark.sql("select tb.TimeBucket, qa.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.qatasks_data AS qa on qa.result_on >= tb.TimeBucket AND (qa.result_on < (tb.TimeBucket + interval 15 minutes)) AND qa.pl_desc = 'TT MP3M' AND (qa.var_id IN (11531 ,    11535 ,    10782 ,    59403 ,    59404 ,    11045 ,    11519 ,    11531 ,    177112 )) GROUP BY qa.var_id , tb.TimeBucket")
m3_qa = sparkdf3mqa.toPandas()
# parent roll id for 3M
sparkdf3mprid = spark.sql("select tb.TimeBucket, result, prod_code FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP3M' AND (cl.var_id IN ('10744'))")
m3_prid = sparkdf3mprid.toPandas()
# historian for 3M
sparkdf3mhis = spark.sql("SELECT tb.TimeBucket, AVG(value_double) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN (SELECT *, GT.ts AS resultOnUTC FROM silver_mfg_ot.gehistorian_fc_timeseries GT WHERE GT.site = 'MP' AND GT.tag_name = 'MK2MIH.CM_2HDBXPR_DACA_PV.F_CV') as gt on gt.resultOnUTC >= tb.TimeBucket AND (gt.resultOnUTC < (tb.TimeBucket + interval 15 minutes)) GROUP BY tb.TimeBucket")
m3_his = sparkdf3mhis.toPandas()
# 3m sideloaded variables
m3_cl1 = m3_cl
m3_qa1 = m3_qa
m3_prid1 = m3_prid
m3_side = side.loc[side['pl_desc'] == 'TT MP3M']
m3_extraprid = m3_side.loc[m3_side['var_id'] == 10744]

# COMMAND ----------

m3_side = side.loc[side['pl_desc'] == 'TT MP3M']
m3_extraprid = m3_side.loc[m3_side['var_id'] == 10744]

# COMMAND ----------

m3_side = m3_side[m3_side.var_id != 10744]
m3_side['result'] = m3_side['result'].astype('float')
m3_side['result_on'] = pd.to_datetime(m3_side['result_on'])
m3_side = m3_side.rename(columns = {'result_on': 'TimeBucket'})
m3_side = m3_side.pivot_table(index = 'TimeBucket', columns = 'var_id', values = 'result')
m3_side = m3_side.reset_index()
m3_side.columns.name = None
m3_side.columns = ['TimeBucket','Creper Life', 'Caliper A Roll','Caliper B Roll']
m3_side['Creper Life'] = m3_side['TimeBucket']
m3_side['TimeBucket'] = m3_side['TimeBucket'].dt.round('15min')
m3_side = m3_side.set_index('TimeBucket')
m3_side = timeframe.join(m3_side, how = 'outer')
m3_side = m3_side.drop(columns = ['Hold'])
m3_side = m3_side.ffill()
m3_side = m3_side.reset_index()
m3_side['Creper Life'] = (m3_side['TimeBucket'] - m3_side['Creper Life'])/pd.Timedelta(minutes = 1)
m3_side['Creper Life'] = [0 if x < 0 else x for x in m3_side['Creper Life']]
m3_side = m3_side.set_index('TimeBucket')

# COMMAND ----------

m3_extraprid['TimeBucket'] = pd.to_datetime(m3_extraprid['result_on'])
m3_extraprid = m3_extraprid.sort_values(by = 'TimeBucket')
m3_extraprid['TimeBucket'] = m3_extraprid['TimeBucket'].dt.round('15min')
m3_extraprid = m3_extraprid.drop(columns = ['result_on','var_id','pl_desc'])
m3_extraprid.columns = ['PRID','TimeBucket']

# COMMAND ----------

m3_cl = m3_cl1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m3_cl = m3_cl.rename(columns = variableListMP3_cl)
m3_cl = m3_cl.reset_index()
m3_cl['TimeBucket'] = pd.to_datetime(m3_cl['TimeBucket'])
m3_cl = m3_cl.sort_values(by = 'TimeBucket')
m3_cl_clean = m3_cl[['TimeBucket','Cleaning Blade Life']]
m3_cl_clean = m3_cl_clean.dropna(subset = ['Cleaning Blade Life'])
m3_cl_clean['Cleaning Blade Life'] = m3_cl_clean['TimeBucket']
m3_cl_clean = m3_cl_clean.set_index('TimeBucket')
m3_cl = m3_cl.drop(columns = ['Cleaning Blade Life'])
m3_cl = m3_cl.set_index('TimeBucket')
m3_cl = m3_cl.join(m3_cl_clean, how = 'outer')
m3_qa = m3_qa1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m3_qa = m3_qa.rename(columns = variableListMP3_qa)
m3_qa = m3_qa.reset_index()
m3_qa['TimeBucket'] = pd.to_datetime(m3_qa['TimeBucket'])
m3_qa = m3_qa.sort_values(by = 'TimeBucket')
m3_qa = m3_qa.set_index('TimeBucket')
m3_prid['TimeBucket'] = pd.to_datetime(m3_prid['TimeBucket'])
m3_prid = m3_prid.sort_values(by = 'TimeBucket')
m3_prid.columns = ['TimeBucket','PRID','Product Code']
m3_prid2 = m3_prid
m3_prid2 = m3_prid2.drop(columns = ['Product Code'])
m3_prid2 = pd.concat([m3_extraprid,m3_prid2])
mp3m = timeframe.join(m3_cl)
mp3m = mp3m.join(m3_qa)
mp3m = mp3m.drop(columns = ['Hold'])
mp3m = mp3m.ffill()
mp3m = mp3m.drop(columns = ['Tensile Modulus CD'])
m3_side = m3_side.drop(columns = ['Caliper A Roll','Caliper B Roll'])
mp3m = mp3m.join(m3_side, how = 'outer')
mp3m2 = mp3m.reset_index()
mp3m2['Cleaning Blade Life'] = mp3m2['TimeBucket'] - mp3m2['Cleaning Blade Life']
mp3m2['Cleaning Blade Life'] = mp3m2['Cleaning Blade Life']/pd.Timedelta(minutes = 1)

# COMMAND ----------

# DBTITLE 1,4M
# 4M centerline variables
sparkdf4mcl = spark.sql("select tb.TimeBucket, cl.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP4M' AND (cl.var_id IN ('170734','167951','32845','167962','167963','167965','32266','32268','32220','32356','32539','32651','32928','32687','32314','32305','32409','32389','66470')) GROUP BY cl.var_id , tb.TimeBucket")
m4_cl = sparkdf4mcl.toPandas()
# 4M quality variables
sparkdf4mqa = spark.sql("select tb.TimeBucket, qa.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.qatasks_data AS qa on qa.result_on >= tb.TimeBucket AND (qa.result_on < (tb.TimeBucket + interval 15 minutes)) AND qa.pl_desc = 'TT MP4M' AND (qa.var_id IN ('33851','33857','33876','33788','59427','59428','33432','33860','33869','72938')) GROUP BY qa.var_id , tb.TimeBucket")
m4_qa = sparkdf4mqa.toPandas()
# parent roll id for 4M
sparkdf4mprid = spark.sql("select tb.TimeBucket, result, prod_code FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-20'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP4M' AND (cl.var_id IN ('33179'))")
m4_prid = sparkdf4mprid.toPandas()
# historian for 4M
# 4m sideloaded variables
m4_side = side.loc[side['pl_desc'] == 'TT MP4M']
m4_extraprid = m4_side.loc[m4_side['var_id'] == 33179]
m4_cl1 = m4_cl
m4_qa1 = m4_qa
m4_prid1 = m4_prid

# COMMAND ----------

m4_side = side.loc[side['pl_desc'] == 'TT MP4M']
m4_extraprid = m4_side.loc[m4_side['var_id'] == 33179]

# COMMAND ----------

m4_side = m4_side[m4_side.var_id != 33179]
m4_side['result'] = m4_side['result'].astype('float')
m4_side['result_on'] = pd.to_datetime(m4_side['result_on'])
m4_side = m4_side.rename(columns = {'result_on': 'TimeBucket'})
m4_side = m4_side.pivot_table(index = 'TimeBucket', columns = 'var_id', values = 'result')
m4_side = m4_side.reset_index()
m4_side.columns.name = None
m4_side.columns = ['TimeBucket','Creper Life', 'Caliper A Roll','Caliper B Roll']
m4_side['Creper Life'] = m4_side['TimeBucket']
m4_side['TimeBucket'] = m4_side['TimeBucket'].dt.round('15min')
m4_side = m4_side.set_index('TimeBucket')
m4_side = timeframe.join(m4_side, how = 'outer')
m4_side = m4_side.drop(columns = ['Hold'])
m4_side = m4_side.ffill()
m4_side = m4_side.reset_index()
m4_side['Creper Life'] = (m4_side['TimeBucket'] - m4_side['Creper Life'])/pd.Timedelta(minutes = 1)
m4_side['Creper Life'] = [0 if x < 0 else x for x in m4_side['Creper Life']]
m4_side = m4_side.set_index('TimeBucket')

# COMMAND ----------

m4_extraprid['TimeBucket'] = pd.to_datetime(m4_extraprid['result_on'])
m4_extraprid = m4_extraprid.sort_values(by = 'TimeBucket')
m4_extraprid['TimeBucket'] = m4_extraprid['TimeBucket'].dt.round('15min')
m4_extraprid = m4_extraprid.drop(columns = ['result_on','var_id','pl_desc'])
m4_extraprid.columns = ['PRID','TimeBucket']

# COMMAND ----------

m4_cl = m4_cl1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m4_cl = m4_cl.rename(columns = variableListMP4_cl)
m4_cl = m4_cl.reset_index()
m4_cl['TimeBucket'] = pd.to_datetime(m4_cl['TimeBucket'])
m4_cl = m4_cl.sort_values(by = 'TimeBucket')
m4_cl_clean = m4_cl[['TimeBucket','Cleaning Blade Life']]
m4_cl_clean = m4_cl_clean.dropna(subset = ['Cleaning Blade Life'])
m4_cl_clean['Cleaning Blade Life'] = m4_cl_clean['TimeBucket']
m4_cl_clean = m4_cl_clean.set_index('TimeBucket')
m4_cl = m4_cl.drop(columns = ['Cleaning Blade Life'])
m4_cl = m4_cl.set_index('TimeBucket')
m4_cl = m4_cl.join(m4_cl_clean, how = 'outer')
m4_qa = m4_qa1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m4_qa = m4_qa.rename(columns = variableListMP4_qa)
m4_qa = m4_qa.reset_index()
m4_qa['TimeBucket'] = pd.to_datetime(m4_qa['TimeBucket'])
m4_qa = m4_qa.sort_values(by = 'TimeBucket')
m4_qa = m4_qa.set_index('TimeBucket')
m4_prid['TimeBucket'] = pd.to_datetime(m4_prid['TimeBucket'])
m4_prid = m4_prid.sort_values(by = 'TimeBucket')
m4_prid.columns = ['TimeBucket','PRID','Product Code']
m4_prid2 = m4_prid
m4_prid2 = m4_prid2.drop(columns = ['Product Code'])
m4_prid2 = pd.concat([m4_extraprid,m4_prid2])
mp4m = timeframe.join(m4_cl)
mp4m = mp4m.join(m4_qa)
mp4m = mp4m.drop(columns = ['Hold'])
mp4m = mp4m.ffill()
mp4m = mp4m.drop(columns = ['Work to Tear', 'Tensile Modulus CD'])
m4_side = m4_side.drop(columns = ['Caliper A Roll','Caliper B Roll'])
mp4m = mp4m.join(m4_side, how = 'outer')
mp4m2 = mp4m.reset_index()
mp4m2['Cleaning Blade Life'] = mp4m2['TimeBucket'] - mp4m2['Cleaning Blade Life']
mp4m2['Cleaning Blade Life'] = mp4m2['Cleaning Blade Life']/pd.Timedelta(minutes = 1)

# COMMAND ----------

# DBTITLE 1,7M
# 7M centerline variables
sparkdf7mcl = spark.sql("select tb.TimeBucket, cl.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-19'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP7M' AND (cl.var_id IN ('170762','65534','22462','173727','173726','173729','170759','170761','21578','21914','22116','22221','22509','22237','21754','21739','22023','22003','66470')) GROUP BY cl.var_id , tb.TimeBucket")
m7_cl = sparkdf7mcl.toPandas()
# 7M quality variables
sparkdf7mqa = spark.sql("select tb.TimeBucket, qa.var_id, AVG(result) FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-19'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.qatasks_data AS qa on qa.result_on >= tb.TimeBucket AND (qa.result_on < (tb.TimeBucket + interval 15 minutes)) AND qa.pl_desc = 'TT MP7M' AND (qa.var_id IN ('23512','23540','23545','23400','59499','59500','62443','23767','23536','73083')) GROUP BY qa.var_id , tb.TimeBucket")
m7_qa = sparkdf7mqa.toPandas()
# parent roll id for 7M
sparkdf7mprid = spark.sql("select tb.TimeBucket, result, prod_code FROM (SELECT explode(sequence(to_timestamp('2022-01-01'), to_timestamp('2023-07-19'), interval 15 minutes)) AS TimeBucket) tb JOIN silver_iods.cltasks_data AS cl on cl.result_on >= tb.TimeBucket AND (cl.result_on < (tb.TimeBucket + interval 15 minutes)) AND cl.pl_desc = 'TT MP7M' AND (cl.var_id IN ('22793'))")
m7_prid = sparkdf7mprid.toPandas()
# historian for 7M
# 7m sideloaded variables
m7_side = side.loc[side['pl_desc'] == 'TT MP7M']
m7_extraprid = m7_side.loc[m7_side['var_id'] == 22793]
m7_cl1 = m7_cl
m7_qa1 = m7_qa
m7_prid1 = m7_prid

# COMMAND ----------

m7_side = side.loc[side['pl_desc'] == 'TT MP7M']
m7_extraprid = m7_side.loc[m7_side['var_id'] == 22793]

# COMMAND ----------

m7_side = m7_side[m7_side.var_id != 22793]
m7_side['result'] = m7_side['result'].astype('float')
m7_side['result_on'] = pd.to_datetime(m7_side['result_on'])
m7_side = m7_side.rename(columns = {'result_on': 'TimeBucket'})
m7_side = m7_side.pivot_table(index = 'TimeBucket', columns = 'var_id', values = 'result')
m7_side = m7_side.rename(columns = {59499: 'Caliper A Roll', 59500:'Caliper B Roll', 22023:'Creper Life'})
m7_side = m7_side.reset_index()
m7_side.columns.name = None
m7_side['Creper Life'] = m7_side['TimeBucket']
m7_side['TimeBucket'] = m7_side['TimeBucket'].dt.round('15min')
m7_side = m7_side.set_index('TimeBucket')
m7_side = timeframe.join(m7_side, how = 'outer')
m7_side = m7_side.ffill()
m7_side = m7_side.reset_index()
m7_side['Creper Life'] = (m7_side['TimeBucket'] - m7_side['Creper Life'])/pd.Timedelta(minutes = 1)
m7_side['Creper Life'] = [0 if x < 0 else x for x in m7_side['Creper Life']]
m7_side = m7_side.drop(columns = 'Hold')
m7_side = m7_side.set_index('TimeBucket')

# COMMAND ----------

m7_extraprid['TimeBucket'] = pd.to_datetime(m7_extraprid['result_on'])
m7_extraprid = m7_extraprid.sort_values(by = 'TimeBucket')
m7_extraprid['TimeBucket'] = m7_extraprid['TimeBucket'].dt.round('15min')
m7_extraprid = m7_extraprid.drop(columns = ['result_on','var_id','pl_desc'])
m7_extraprid.columns = ['PRID','TimeBucket']

# COMMAND ----------

m7_cl = m7_cl1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m7_cl = m7_cl.rename(columns = variableListMP7_cl)
m7_cl = m7_cl.reset_index()
m7_cl['TimeBucket'] = pd.to_datetime(m7_cl['TimeBucket'])
m7_cl = m7_cl.sort_values(by = 'TimeBucket')
m7_cl_clean = m7_cl[['TimeBucket','Cleaning Blade Life']]
m7_cl_clean = m7_cl_clean.dropna(subset = ['Cleaning Blade Life'])
m7_cl_clean['Cleaning Blade Life'] = m7_cl_clean['TimeBucket']
m7_cl_clean = m7_cl_clean.set_index('TimeBucket')
m7_cl = m7_cl.drop(columns = ['Cleaning Blade Life'])
m7_cl = m7_cl.set_index('TimeBucket')
m7_cl = m7_cl.join(m7_cl_clean, how = 'outer')
m7_qa = m7_qa1.pivot(index = 'TimeBucket',columns = 'var_id', values = 'avg(result)')
m7_qa = m7_qa.rename(columns = variableListMP7_qa)
m7_qa = m7_qa.reset_index()
m7_qa['TimeBucket'] = pd.to_datetime(m7_qa['TimeBucket'])
m7_qa = m7_qa.sort_values(by = 'TimeBucket')
m7_qa = m7_qa.set_index('TimeBucket')
m7_prid['TimeBucket'] = pd.to_datetime(m7_prid['TimeBucket'])
m7_prid = m7_prid.sort_values(by = 'TimeBucket')
m7_prid.columns = ['TimeBucket','PRID','Product Code']
mp7m = timeframe.join(m7_cl)
mp7m = mp7m.join(m7_qa)
mp7m = mp7m.drop(columns = ['Hold'])
mp7m = mp7m.ffill()
mp7m = mp7m.drop(columns = ['Work to Tear', 'Tensile Modulus CD'])
mp7m = mp7m.join(m7_side, how = 'outer')

# COMMAND ----------

mp7m2 = mp7m.reset_index()
mp7m2['Cleaning Blade Life'] = mp7m2['TimeBucket'] - mp7m2['Cleaning Blade Life']
mp7m2['Cleaning Blade Life'] = mp7m2['Cleaning Blade Life']/pd.Timedelta(minutes = 1)

# COMMAND ----------

m7_prid2 = m7_prid
m7_prid2 = m7_prid2.drop(columns = ['Product Code'])
m7_prid2 = pd.concat([m7_extraprid,m7_prid2])

# COMMAND ----------

sparkdf = spark.sql("select * from groupdb_famc_qa.humidity_data_mehoop")
humidity = sparkdf.toPandas()

# COMMAND ----------

humidity['Date'] = pd.to_datetime(humidity['Date'])
humidity['Max'] = humidity['Max'].astype('float')
humidity['Min'] = humidity['Min'].astype('float')
humidity['Avg'] = humidity['Avg'].astype('float')
humidity.columns = ['TimeBucket', 'Max','Avg','Min']
humidity = humidity.set_index('TimeBucket')
humidity = timeframe.join(humidity)
humidity = humidity.ffill()
humidity = humidity.drop(columns = ['Hold'])
humidity = humidity.reset_index()

# COMMAND ----------

# DBTITLE 1,Joining Paper and Converting
mp7m_pr = mp7m2.merge(m7_prid2, on = 'TimeBucket', how = 'outer')
mp7m_pr = mp7m_pr.rename(columns = {'TimeBucket' : 'PRID Made'})
mp7m_pr['Paper Machine'] = '7M'
mp1m_pr = mp1m2.merge(m1_prid2, on = 'TimeBucket', how = 'outer')
mp1m_pr = mp1m_pr.rename(columns = {'TimeBucket' : 'PRID Made', 'PRID_y': 'PRID'})
mp1m_pr = mp1m_pr.drop(columns = ['PRID_x'])
mp1m_pr['Paper Machine'] = '1M'
mp2m_pr = mp2m2.merge(m2_prid2, on = 'TimeBucket', how = 'outer')
mp2m_pr = mp2m_pr.rename(columns = {'TimeBucket' : 'PRID Made', 'PRID_y': 'PRID'})
mp2m_pr['Paper Machine'] = '2M'
mp3m_pr = mp3m2.merge(m3_prid2, on = 'TimeBucket', how = 'outer')
mp3m_pr = mp3m_pr.rename(columns = {'TimeBucket' : 'PRID Made', 'PRID_y': 'PRID'})
mp3m_pr['Paper Machine'] = '3M'
mp4m_pr = mp4m2.merge(m4_prid2, on = 'TimeBucket', how = 'outer')
mp4m_pr = mp4m_pr.rename(columns = {'TimeBucket' : 'PRID Made', 'PRID_y': 'PRID'})
mp4m_pr['Paper Machine'] = '4M'
paper = pd.concat([mp1m_pr,mp2m_pr,mp3m_pr,mp4m_pr,mp7m_pr])
paper['% Unrefined'] = 100 - paper['% Refined'] - paper['% Broke']
paper = paper.dropna(subset = ['PRID'])
mk81_2 = mk81.reset_index()
join_data = mk81_2.merge(paper,on = 'PRID', how = 'outer')
join_data = join_data.drop(columns = ['Caliper Range'])
join_data['Paper Storage Time'] = (join_data['TimeBucket'] - join_data['PRID Made']) / pd.Timedelta(minutes = 1)

# COMMAND ----------

categ_var = join_data[['Paper Machine']]
categ_var = pd.get_dummies(categ_var)
categ_var.columns = ['1M','2M','3M','4M','7M']
join_data = join_data.join(categ_var)
join_data = join_data.drop(columns = ['PRID','Paper Machine'])
join_data[['Average Emboss',  'Line Speed', 'Rt Load Cell', 'Lt Load Cell',
       'PVA Flow', 'Winder Tension', 'UWS Normal Tension', 'Online Glue Temp',
       'Recal. Gap DS', 'Recal. Gap OS', 'App. Gap DS', 'App. Gap OS',
       'Cleaned', 'Anilox Nip OS', 'Anilox Nip Center', 'Anilox Nip DS',
       'Glue % Solids', 'SER/App. Gap OS', 'SER/App. Gap Center',
       'SER/App. Gap DS', 'Marrying Nip OS', 'Marrying Nip Center',
       'Marrying Nip DS', 'Pressure Nip OS', 'Pressure Nip Center',
       'Pressure Nip DS', 'Marrying Nip Avg', 'Pressure Nip Avg',
       'SER/App Gap Average',  'Headbox Pressure',
       'Wire to Press Draw', 'Reel Moisture', 'PUS Vacuum', 'Reel Speed',
       'Yankee Speed', 'Absorb Aid', '% SSK', 'CMC', 'Emulsion', 'Kymene',
       'Cleaning Blade Life', '% Refined', '% NSK', '% Euc', '% Broke',
       'Yankee to Reel Draw', 'Basis Weight', 'Stretch', 'Tensile Ratio',
       'Tensiles', 'Wet Burst', 'Caliper A Roll', 'Caliper B Roll',
       'Creper Life',  '% Unrefined']]= join_data[['Average Emboss',  'Line Speed', 'Rt Load Cell', 'Lt Load Cell',
       'PVA Flow', 'Winder Tension', 'UWS Normal Tension', 'Online Glue Temp',
       'Recal. Gap DS', 'Recal. Gap OS', 'App. Gap DS', 'App. Gap OS',
       'Cleaned', 'Anilox Nip OS', 'Anilox Nip Center', 'Anilox Nip DS',
       'Glue % Solids', 'SER/App. Gap OS', 'SER/App. Gap Center',
       'SER/App. Gap DS', 'Marrying Nip OS', 'Marrying Nip Center',
       'Marrying Nip DS', 'Pressure Nip OS', 'Pressure Nip Center',
       'Pressure Nip DS', 'Marrying Nip Avg', 'Pressure Nip Avg',
       'SER/App Gap Average',  'Headbox Pressure',
       'Wire to Press Draw', 'Reel Moisture', 'PUS Vacuum', 'Reel Speed',
       'Yankee Speed', 'Absorb Aid', '% SSK', 'CMC', 'Emulsion', 'Kymene',
       'Cleaning Blade Life', '% Refined', '% NSK', '% Euc', '% Broke',
       'Yankee to Reel Draw', 'Basis Weight', 'Stretch', 'Tensile Ratio',
       'Tensiles', 'Wet Burst', 'Caliper A Roll', 'Caliper B Roll',
       'Creper Life',  '% Unrefined']].astype('float')

# COMMAND ----------

# DBTITLE 1,Trying to create new variable for how much is built up on the E/L
join_data['Build Up'] = [x/30 for x in join_data['% Euc']]
join_data['Build Up3'] = [4 if x > 40 else 1 for x in join_data['% Euc']]
join_data['Build Up2'] = [1 if x > 15 else 0 for x in join_data['Cleaned']]
join_data['Storage Factor'] = [2 if x > 840 else 1 for x in join_data['Paper Storage Time']]
join_data['7 Factor'] = [2 if x == 1 else 1 for x in join_data['7M']]
join_data['Refined Factor'] = [x/52 for x in join_data['% Refined']]
join_data['Broke Factor'] = [x/10 for x in join_data['% Broke']]
join_data['Built1'] = join_data['% Euc'] * join_data['Build Up2']  * join_data['Build Up3'] * join_data['Storage Factor']
join_data['Built'] = join_data.groupby(join_data['Built1'].eq(0).cumsum()).cumcount()
join_data = join_data.dropna()

# COMMAND ----------

join_data = join_data.dropna(subset = ['Plybond Test Average'])
join_data = join_data.drop(columns = ['Build Up','Build Up2','Build Up3','Storage Factor','Built1','7 Factor','Broke Factor','Refined Factor'])
join_data['Wet/Dry Strength'] = join_data['Kymene'] / join_data['CMC']
join_data['Rolls Warm'] = [1 if x > 20 else 0 for x in join_data['Cleaned']]
join_data['Viscosity'] = (join_data['PVA Flow']*join_data['Glue % Solids']) / join_data['Online Glue Temp']
join_data['Caliper Average'] = (join_data['Caliper A Roll'] + join_data['Caliper B Roll'])/2
join_data['Paper Storage Time'] = [np.nan if x < 0 else x for x in join_data['Paper Storage Time']]
join_data = join_data.dropna()

# COMMAND ----------

# DBTITLE 1,Code to try to create a humidity impact term
result = []
for index, row in join_data.iterrows():
    start_date = row['PRID Made']
    end_date = row['TimeBucket']

    filt = humidity[(humidity['TimeBucket'] >= start_date) & (humidity['TimeBucket'] <= end_date)]
    sum_over = filt[filt['Avg'] > 64]
    sum_values = sum_over['Max'].sum() 
    sum_values = row['Caliper Average'] * sum_values / 10000 + row['Wet Burst'] * sum_values / 10000 + row['Reel Moisture'] * sum_values / 10000
    result.append(sum_values)
join_data['Humidity'] = result

# COMMAND ----------

validation = join_data.mask(join_data['TimeBucket'] < '2023-06-01', np.nan)
join_data = join_data.mask(join_data['TimeBucket'] > '2023-06-01', np.nan)
validation = validation.dropna()
join_data = join_data.dropna()
join_data = join_data.groupby('TimeBucket').mean()
validation = validation.groupby('TimeBucket').mean()

# COMMAND ----------

# DBTITLE 1,Working with Predefined Table
all_1 = pd.concat([join_data,validation])
sql_all = all_1.to_sql(name = 'MK81_Plybond_Features', con = )

# COMMAND ----------

fig = plt.figure(figsize = (30,30))
plt.plot(jda['Plybond Test Average'], 'b*', markersize = 30)
plt.axhline(y = 4, color = 'r', linestyle= '--', linewidth = 8)
plt.axhline(y = 7, color = 'g', linestyle= '--', linewidth = 8)
plt.axhline(y = 18, color = 'k', linestyle= '--', linewidth = 8)
plt.title('Plybond Test Average', fontsize = 50)
plt.xlabel('Time', fontsize = 50)
plt.ylabel('Test Result (g/in)', fontsize = 50)
plt.xticks(rotation = 90, fontsize = 50)
plt.yticks(fontsize = 50)

# COMMAND ----------

sns.pairplot(join_data, y_vars = ['Plybond Test Average'])

# COMMAND ----------

join_data['Product Quality'] = [1 if x > 11 else 0 for x in join_data['Plybond Test Average']]
validation['Product Quality'] = [1 if x > 11 else 0 for x in validation['Plybond Test Average']]

# COMMAND ----------

join_data2 = join_data
#join_data2 = join_data2.mask(join_data2['Line Speed'] < 450, np.nan)
#join_data2 = join_data2.mask(join_data2['Reel Speed'] < 2000, np.nan)
join_data2 = join_data2.dropna()
#join_data2 = join_data2.drop(columns = ['Glue % Solids','PVA Flow','Humidity','% Refined','% Broke'])
join_data2 = join_data2[['Plybond Test Average','Standard Deviation','Product Quality','Paper Storage Time','% Unrefined','Viscosity','Average Emboss','Cleaned']]

# COMMAND ----------

validation2 = validation
#validation2 = validation2.mask(validation2['Line Speed'] < 450, np.nan)
#validation2 = validation2.mask(validation2['Reel Speed'] < 2000, np.nan)
validation2 = validation2.dropna()
validation2 = validation2[['Plybond Test Average','Standard Deviation','Product Quality', 'Paper Storage Time','% Unrefined','Viscosity','Average Emboss','Cleaned']]

# COMMAND ----------

#'Paper Storage Time','% Unrefined','Viscosity','Average Emboss','Cleaned'

# COMMAND ----------

x_val = validation2.drop(['Plybond Test Average','Product Quality', 'Standard Deviation'], axis = 1)
y_val = validation2['Plybond Test Average']
val_qual = validation2['Product Quality']

# COMMAND ----------

train, test = train_test_split(join_data2, test_size = 0.1, shuffle = False)
x_train = train.drop(['Plybond Test Average','Product Quality', 'Standard Deviation'], axis = 1)
x_test = test.drop(['Plybond Test Average','Product Quality', 'Standard Deviation'], axis = 1)
y_train = train['Plybond Test Average']
y_test = test['Plybond Test Average']
train_qual = train['Product Quality']
test_qual = test['Product Quality']

# COMMAND ----------

over_sample = RandomOverSampler(sampling_strategy = 'not majority')
x_over, y_over = over_sample.fit_resample(x_train, train_qual)
under_sample = RandomUnderSampler(sampling_strategy = 'not majority')
x_under, y_under = under_sample.fit_resample(x_train, train_qual)
smote = SMOTE(sampling_strategy = 'not majority')
x_smote, y_smote = smote.fit_resample(x_train, train_qual)
tomeklinks = TomekLinks(sampling_strategy = 'majority')
x_tl, y_tl = tomeklinks.fit_resample(x_train, train_qual)

# COMMAND ----------

sel = SelectFromModel(GradientBoostingClassifier(), max_features = 6)
sel.fit(x_val, val_qual)
selected_features = x_val.columns[(sel.get_support())]
print(selected_features)

# COMMAND ----------

gb_class = GradientBoostingClassifier(max_depth = 2, n_estimators = 50, min_samples_leaf = 5, min_samples_split = 5).fit(x_under, y_under)
gb_pred = gb_class.predict(x_test)
gb_score = gb_class.score(x_test, test_qual)
false_positive_rate, true_positive_rate, thresholds = roc_curve(test_qual, gb_pred)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('Score:', gb_score)

# COMMAND ----------

y_test2 = test_qual.reset_index()
plt.plot(y_test2['Product Quality'], 'b*-')
plt.plot(gb_pred,'ro')
plt.title('GB Plybond')
plt.ylabel('Plybond Test Average (g/in)')

# COMMAND ----------

gb_pred2 = pd.DataFrame(gb_pred)
pred = y_test2.join(gb_pred2)
pred = pred.set_index('TimeBucket')
pred.columns = ['Product Quality','GB Pred']
y_test2 = pd.DataFrame(y_test)
pred = pred.join(y_test2)
pred['Dev'] = pred['Product Quality'] - pred['GB Pred']
pred = pred.reset_index()
len(pred[np.abs(pred['Plybond Test Average']) <=11])

# COMMAND ----------

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

# COMMAND ----------

color_labels = [0,1]
corgb_values = sns.color_palette('Set1', 4)
color_map = dict(zip(color_labels, corgb_values))
pred2 = pred.reset_index()
scatter = plt.scatter(pred2['TimeBucket'],pred2['Plybond Test Average'], color = pred2['GB Pred'].map(color_map))
plt.axhline(y = 7, color = 'k', linestyle = '--')
plt.axhline(y = 9, color = 'r', linestyle = '--')
plt.axhline(y = 11, color = 'g', linestyle = '--')
plt.axhline(y = 13, color = 'r', linestyle = '--')
plt.xlabel('Time Bucket')
plt.xticks(rotation = 90)
plt.ylabel('Plybond Test Average')
plt.title('Plybond Test Average with Gradient Boosting')

# COMMAND ----------

pred3 = pred[(pred['Plybond Test Average'] < 9) |(pred['Plybond Test Average'] > 13)]

# COMMAND ----------

matrix = metrics.confusion_matrix(pred3['Product Quality'],pred3['GB Pred'])
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
matrix = np.round(matrix, decimals = 2)
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':40},
            cmap=plt.cm.Greens, linewidths=0.2)
class_names = ['Bad','Good']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25, fontsize = 30)
plt.yticks(tick_marks2, class_names, rotation=0, fontsize = 30)
plt.xlabel('Predicted label', fontsize = 50)
plt.ylabel('True label', fontsize = 50)
plt.title('Confusion Matrix',fontsize = 50)
plt.show()

# COMMAND ----------

# DBTITLE 1,Target Shuffling
def target_shuffling(target):
    shuffled_target = np.copy(target)
    np.random.shuffle(shuffled_target)
    return shuffled_target

# COMMAND ----------

num_simulations = 1000
results = []
original_metric = None

for i in range(num_simulations):
    if original_metric == None:
        model = GradientBoostingClassifier(n_estimators = 100, max_depth = 2,random_state = 32).fit(x_under, y_under)
        prediction = model.predict(x_test)
        false_positive_rate, true_positive_rate, thresholds = roc_curve(test_qual, prediction)
        roc_auc = auc(false_positive_rate, true_positive_rate)
        original_metric = roc_auc
    
    shuffled_target = target_shuffling(y_under)

    model = GradientBoostingClassifier(n_estimators = 100, max_depth = 2,random_state = 32).fit(x_under, shuffled_target)
    prediction = model.predict(x_test)
    false_positive_rate, true_positive_rate, thresholds = roc_curve(test_qual, prediction)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    results.append(roc_auc)

num_better_than_original = np.sum(result >= original_metric for result in results)
probability_of_chance = num_better_than_original / num_simulations
print("Probability of success due to chance:", probability_of_chance)

# COMMAND ----------

# DBTITLE 1,Validation
gb_val = gb_class.predict(x_val)
gb_valscore = gb_class.score(x_val, val_qual)
false_positive_rate, true_positive_rate, thresholds = roc_curve(val_qual, gb_val)
roc_auc = auc(false_positive_rate, true_positive_rate)
print('Score:', roc_auc)

# COMMAND ----------

gb_pred2 = pd.DataFrame(gb_val)
y_test2 = pd.DataFrame(y_val)
y_test2 = y_test2.reset_index()
pred = y_test2.join(gb_pred2)
pred.columns = ['TimeBucket','Plybond Test Average','Validation Quality']
val_qual2 = pd.DataFrame(val_qual)
pred = pred.set_index('TimeBucket')
pred = pred.join(val_qual)
pred['Dev'] = pred['Product Quality'] - pred['Validation Quality']
color_labels = [0,1]
corgb_values = sns.color_palette('Set1', 4)
color_map = dict(zip(color_labels, corgb_values))
pred2 = pred.reset_index()
scatter = plt.scatter(pred2['TimeBucket'],pred2['Plybond Test Average'], color = pred2['Validation Quality'].map(color_map))
plt.axhline(y = 7, color = 'k', linestyle = '--')
plt.axhline(y = 9, color = 'r', linestyle = '--')
plt.axhline(y = 11, color = 'g', linestyle = '--')
plt.axhline(y = 13, color = 'r', linestyle = '--')
plt.xlabel('Time Bucket')
plt.xticks(rotation = 90)
plt.ylabel('Plybond Test Average')
plt.title('Plybond Test Average with Gradient Boosting')

# COMMAND ----------

pred3 = pred[(pred['Plybond Test Average'] < 9) | (pred['Plybond Test Average'] > 13)]
pred3 = pred3.reset_index()
pred3 = pred3[pred3['TimeBucket'] < '2023-06-17']

# COMMAND ----------

matrix = metrics.confusion_matrix(pred3['Product Quality'],pred3['Validation Quality'])
matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]
matrix = np.round(matrix, decimals = 2)
plt.figure(figsize=(16,7))
sns.set(font_scale=1.4)
sns.heatmap(matrix, annot=True, annot_kws={'size':40},
            cmap=plt.cm.Greens, linewidths=0.2)
class_names = ['Bad','Good']
tick_marks = np.arange(len(class_names))
tick_marks2 = tick_marks + 0.5
plt.xticks(tick_marks, class_names, rotation=25, fontsize = 30)
plt.yticks(tick_marks2, class_names, rotation=0, fontsize = 30)
plt.xlabel('Predicted label', fontsize = 50)
plt.ylabel('True label', fontsize = 50)
plt.title('Confusion Matrix',fontsize = 50)
plt.show()

# COMMAND ----------

predv = pred.reset_index()
predv[(np.abs(predv['Plybond Test Average']) < 7 ) ]

# COMMAND ----------

feature_names = gb_class.feature_names_in_
feature_names

# COMMAND ----------

x_train2 = x_under
x_train2 = pd.DataFrame(x_train2)
x_train2 = x_train2.to_numpy()

# COMMAND ----------

explainer = lime.lime_tabular.LimeTabularExplainer(x_train2, feature_names = feature_names,class_names = ['Bad','Good'], mode = 'classification')

# COMMAND ----------

t = train.reset_index()
t[ (t['Viscosity'] > 0.22 ) & (t['Cleaned'] < 360) & (t['Standard Deviation'] < 2) & (t['Paper Storage Time'] > 11565) & (t['Paper Storage Time'] > 345)]
#t[t['Plybond Test Average'] < 7]

# COMMAND ----------

t = test.reset_index()
t[ (t['Plybond Test Average'] < 6 )]

# COMMAND ----------

x = x_train.iloc[[655]]

# COMMAND ----------

point = x
point = point.reset_index()
point = point.drop(columns = ['TimeBucket'])
point = point.to_numpy()
point = point.reshape((5,))

# COMMAND ----------

exp = explainer.explain_instance(point, gb_class.predict_proba)
exp.as_pyplot_figure()

# COMMAND ----------

exp.as_list()

# COMMAND ----------

exp.predict_proba

# COMMAND ----------

explainers = shap.Explainer(gb_class)
shap_values = explainers(x_train)

# COMMAND ----------

shap.plots.bar(shap_values)

# COMMAND ----------

shap.summary_plot(shap_values)

# COMMAND ----------

force_plot = shap.plots.force(explainers.expected_value, shap_values.values, feature_names = x_train.columns)
shap_html = f"{shap.getjs()}{force_plot.html()}"
displayHTML(shap_html)

# COMMAND ----------

shap.dependence_plot("Paper Storage Time", shap_values.values, x_train, interaction_index = "% Unrefined")

# COMMAND ----------

jd = join_data
corr = jd.corr()
corr.style.background_gradient(cmap='coolwarm')

# COMMAND ----------

counts, bins = np.histogram(y_train,bins = [0,5,10,15,20,25,30,35,40])
counts = np.insert(counts, 0, 0)
df = pd.DataFrame(counts, columns = ['Bin Count'])
df['CumSum'] = df['Bin Count'].cumsum()/df['Bin Count'].sum()*100
fig, ax = plt.subplots()
plt.title('Train Plybond Test Average Distribution')
plt.xlabel('Plybond Test Average')
ax.hist(y_train, bins = [0,5,10,15,20,25,30,35,40], weights=np.ones(len(y_train)) / len(y_train), edgecolor='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
ax.grid(False)
ax2 = ax.twinx()
ax2.plot(bins, df['CumSum'], color = 'C1', marker = 'D', ms = 7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.grid(False)

# COMMAND ----------

counts, bins = np.histogram(y_test,bins = [0,5,10,15,20,25,30,35,40])
counts = np.insert(counts, 0, 0)
df = pd.DataFrame(counts, columns = ['Bin Count'])
df['CumSum'] = df['Bin Count'].cumsum()/df['Bin Count'].sum()*100
fig, ax = plt.subplots()
plt.title('Test Plybond Test Average Distribution')
plt.xlabel('Plybond Test Average')
ax.hist(y_test, bins = [0,5,10,15,20,25,30,35,40], weights=np.ones(len(y_test)) / len(y_test), edgecolor='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
ax.grid(False)
ax2 = ax.twinx()
ax2.plot(bins, df['CumSum'], color = 'C1', marker = 'D', ms = 7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.grid(False)

# COMMAND ----------

counts, bins = np.histogram(y_val,bins = [0,5,10,15,20,25,30,35,40])
counts = np.insert(counts, 0, 0)
df = pd.DataFrame(counts, columns = ['Bin Count'])
df['CumSum'] = df['Bin Count'].cumsum()/df['Bin Count'].sum()*100
fig, ax = plt.subplots()
plt.title('Validation Plybond Test Average Distribution')
plt.xlabel('Plybond Test Average')
ax.hist(y_val, bins = [0,5,10,15,20,25,30,35,40], weights=np.ones(len(y_val)) / len(y_val), edgecolor='black')
plt.gca().yaxis.set_major_formatter(PercentFormatter(1))
ax.grid(False)
ax2 = ax.twinx()
ax2.plot(bins, df['CumSum'], color = 'C1', marker = 'D', ms = 7)
ax2.yaxis.set_major_formatter(PercentFormatter())
ax2.grid(False)

# COMMAND ----------

j = join_data
j = j.reset_index()
j = j[j['TimeBucket'] > '2022-07']
j = j.set_index('TimeBucket')

# COMMAND ----------

plt.plot(validation['Average Emboss'],'r*')
plt.plot(j['Average Emboss'],'b*')
plt.xlabel('Time Bucket')
plt.xticks(rotation = 90)
plt.ylabel('Average Emboss')
plt.title('Average Emboss')

# COMMAND ----------


