import xgboost
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split


data=pd.read_excel("loan.xlsx")
data['term'].replace(to_replace=' months', value='', regex=True, inplace=True)
data['term'] = pd.to_numeric(data['term'])
cols=data.columns
for i in cols:
    if data[i].dtype==np.object:
        s=set(data[i])
        d={}
        for i1,j in enumerate(s):
            d[j]=i1
        data[i]=list(map(lambda k:d[k],data[i]))

data.fillna(0,inplace=True)
dt=data[['member_id', 'loan_amnt', 'funded_amnt', 'funded_amnt_inv', 'term', 'int_rate', 'installment', 'sub_grade', 'home_ownership', 'annual_inc', 'loan_status', 'url', 'purpose', 'addr_state', 'dti', 'open_acc', 'revol_bal', 'revol_util', 'total_acc', 'out_prncp', 'out_prncp_inv', 'total_pymnt', 'total_pymnt_inv', 'total_rec_prncp', 'total_rec_int', 'recoveries', 'last_pymnt_amnt']]
x=dt.drop(['loan_status'],1)
y=dt['loan_status']
x=scale(x)
x_tr,x_ts,y_tr,y_ts=train_test_split(x,y,test_size=0.25)

algo = xgboost.sklearn.XGBClassifier(objective="binary:logistic",learning_rate=0.05, 
    seed=9616, 
    max_depth=20, 
    gamma=10, 
    n_estimators=500)

algo.fit(x_tr,y_tr)
print("Acurracy is ",algo.score(x_ts,y_ts))
