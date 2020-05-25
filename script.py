import pandas as pd
import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn import svm
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression,LogisticRegression,ridge_regression
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
df=pd.read_csv('historical_data.csv')
pd.set_option('mode.chained_assignment', None)
#print(df.head())
x1=[]
x1=df.symbol.unique()
print(x1)
def split_date(df):
	df["Date"]=0
	df['Month']=0
	df['Year']=0
	for count,rows in enumerate(df.datetime):
		rows=str(rows)
		df.Month[count]=rows.split("/")[0]
		df.Year[count]=rows.split("/")[-1].split("/")[0]
		df.Date[count]=rows.split("/")[1].split("/")[0]
		#print(date,month,year)
		#df.set_index('datetime')
	return df
#split_date(df)

def get_df(ticker):
	l1=[]
	l2=[]
	l3=[]
	df=pd.read_csv("historical_data.csv")
	for count,symb in enumerate(tqdm(df['symbol'])):
		if ticker==symb:
			l2.append(df['close_price'][count])
			l1.append(df['date_txn'][count])
			l3.append(df['predicted_price'][count])
	return np.array(l1),np.array(l2),np.array(l3)

def create_df(ticker,l1,l2,l3):
	df1=pd.DataFrame(columns=["datetime","predicted_price","close_price"],index=[i for i in range(0,2000)])
	for count,rows in enumerate(l1):
		df1['datetime'][count]=rows
	for count,rows in enumerate(l2):
		df1['close_price'][count]=rows
	for count,rows in enumerate(l3):
		df1['predicted_price'][count]=rows
	#df1.to_csv(f'stock_dfs/{ticker}.csv')
	#df1.set_index('datetime')
	df1.to_csv(f'stock_dfs/{ticker}.csv',index=False)
	return df1
'''l1,l2,l3=get_df('B',df)
df=create_df('B',l1,l2,l3)'''
def make_prediction(ticker):
	'''l1,l2,l3=get_df('B',df)
				df=create_df(l1,l2,l3)'''

	#print(df.head())
	df=pd.read_csv(f"stock_dfs/{ticker}.csv")
	df.dropna(inplace=True)
	x1=[]
	df=split_date(df)
	#df.set_index('datetime')
	df.dropna(inplace=True)
	df.drop(['datetime'],1,inplace=True)
	x=np.array(df.drop(['predicted_price'],1).astype(float))
	#df.set_index(['Date','Month','Year'])
	#x=preprocessing.scale(x)
	#print(x)
	df.reset_index(drop=True, inplace=True)
	y=np.array(df['predicted_price'])
	clf=svm.SVR()
	x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
	clf.fit(x_train,y_train)
	accuracy=clf.score(x_test,y_test)
	print(accuracy)
	df['prediction']=np.NaN
	df['difference_relative']=np.NaN
	df['difference']=np.NaN
	
	for count,idx in enumerate(df.close_price):
		
		Date=df.Date[count]
		Month=df.Month[count]
		df.reset_index(drop=True, inplace=True)
		Year=df.Year[count]

		price=df.close_price[count]
		pred=np.array([price,Date,Month,Year])
		pred=pred.reshape(-1,len(pred))
		df['prediction'][count]=clf.predict(pred)[0]
		df['difference'][count]=(df['prediction'][count]-df['close_price'][count])
		df['difference_relative'][count]=df['difference'][count]/df['close_price'][count]
	#print(df[['prediction','difference','difference_relative']])
	for i in df['difference_relative']:
		x1.append(i)
	#print(x1)
	med=np.median(np.array(x1))
	return med
lister=[]
for i in tqdm(x1):
	print("\n",i)
	medial=make_prediction(i)
	print(medial)
	lister.append(medial)
print(lister)
print('Mean={}'.format(np.mean(np.array(lister))))

