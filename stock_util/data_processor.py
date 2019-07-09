import pandas as pd
import numpy as np
import subprocess
import datetime

SPLUNK_BIN="/Applications/Splunk/bin/splunk"
SPLUNK_OPTIONS=""

class DataLoader():
	def __init__(self):
		self.query=""
		self.df=[]
		self.bid_df=[]
		self.ask_df=[]
		self.starts=[]
		self.forecast=45
		self.window=10
		self.emb_size=10
		self.trading_days=7
		self.ends=[]
		self.day_size=780
		self.bid_query=""
		self.ask_query=""
		self.training=1

	def set_queries(self,bidq, askq):
		self.bid_query=bidq
		self.ask_query=askq

	def set_training(self,training):
		self.training=training

	def get_days(self,length):
		bids=[]
		asks=[]

		bid_df=self.get_splunk_quotes(self.bid_query)
		ask_df=self.get_splunk_quotes(self.ask_query)
		self.set_start_ends(bid_df)

		for i in range(0,len(self.starts)):
			bids.append(bid_df[self.starts[i]:self.ends[i]])
			asks.append(ask_df[self.starts[i]:self.ends[i]])

		asks=pd.concat(asks)
		bids=pd.concat(bids)

		timecols = ['Hour','Minute','Second']
		bids.drop(timecols, inplace=True, axis=1)
		asks.drop(timecols, inplace=True, axis=1)

		df=pd.merge(asks,bids, on="Datetime")
		df=df[:length]
		#df = df.rename(columns={ df.columns[0] : 'Datetime', df.columns[1] : 'Open_x', df.columns[2] : 'High_x', df.columns[3] : 'Low_x', df.columns[4]: 'Close_x', df.columns[5] : 'Volume_x', df.columns[6] : 'Open_y', df.columns[7] : 'High_y', df.columns[8] : 'Low_y', df.columns[9]: 'Close_y', df.columns[10] : 'Volume_y' })
		df = df.rename(columns={ df.columns[0] : 'Datetime', df.columns[1] : 'Open_x', df.columns[2] : 'High_x', df.columns[3] : 'Low_x', df.columns[4]: 'Close_x', df.columns[5] : 'Volume_x', df.columns[6]:  'Weekday_x'})
		df = df.rename( columns={ df.columns[7] : 'Open_y', df.columns[8] : 'High_y', df.columns[9] : 'Low_y', df.columns[10]: 'Close_y', df.columns[11] : 'Volume_y', df.columns[12]: 'Weekday_y' })

		self.df=df

		return df

	def set_start_ends(self,df):
		self.starts,self.ends = [],[] 
		self.starts=self._find_hour(df,'00:00:00')
		self.ends=self._find_hour(df,'16:59:00')

		# Sometimes data come in with the start first. In that case swap
		if(self.ends[0] < self.starts[0]):
			ends=self.ends;
			self.ends=self.starts
			self.starts=ends

		# in case the data ends mid-day
		if(len(self.starts) > len(self.ends)):
			del self.starts[len(self.ends)-1:]

	## Training and test data routines. Returns np arrays for fit
	## These can be used for any df that fits i.e. doesn't re-query datasource
	def get_train_test(self,df,percentage,step=1):

		X, Y = self.get_tensors(df,step);
		X, Y = np.array(X), np.array(Y)
		print("Tensor X/Y len: ",len(X)," / ",len(Y))

		#X_train, X_test, y_train, y_test = self._train_test(X[:-self.day_size*self.trading_days], Y[:-self.day_size*self.trading_days],percentage)
		X_train, X_test, y_train, y_test = self._train_test(X,Y,percentage)
		X_train, X_test = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], self.emb_size)), np.reshape(X_test, (X_test.shape[0], X_test.shape[1], self.emb_size))

		print("\nTesting dataset dimension: [", X_test.shape[0],"/",X_train.shape[0],"]\n")

		return X_train, X_test, y_train, y_test

	def _train_test(self,X, y, percentage):
		p = int(len(X) * percentage)
		X_train = X[0:p]
		Y_train = y[0:p]
 
		X_train, Y_train = self._shuffle(X_train, Y_train)
 
		X_test = X[p:]
		Y_test = y[p:]

		return X_train, X_test, Y_train, Y_test

	def _find_hour(self,df,hour='07:00:00'):
		vec=[]
		for i in range(0,len(df.Datetime)):
			currhour="%02d:%02d:%02d" % (df.Hour[i],df.Minute[i],df.Second[i])
			if(currhour == hour):
				vec.append(i)

		return vec

	def _shuffle(self,a,b):
		# courtsey http://stackoverflow.com/users/190280/josh-bleecher-snyder
		# shuffling of training data
		assert len(a) == len(b)
		shuffled_a = np.empty(a.shape, dtype=a.dtype)
		shuffled_b = np.empty(b.shape, dtype=b.dtype)
		permutation = np.random.permutation(len(a))

		for old_index, new_index in enumerate(permutation):
			shuffled_a[new_index] = a[old_index]
			shuffled_b[new_index] = b[old_index]
		return shuffled_a, shuffled_b

	def get_slice(self,start,end):
		return self.df[start:end]

	def get_tensors(self,df,STEP=1):
		count=0
		X,Y = [],[]

		Time = df.Datetime
		aO = df.Open_x.tolist()
		aH = df.High_x.tolist()
		aL = df.Low_x.tolist()
		aC = df.Close_x.tolist()
		aV = df.Volume_x.tolist()
		bO = df.Open_y.tolist()
		bH = df.High_y.tolist()
		bL = df.Low_y.tolist()
		bC = df.Close_y.tolist()
		bV = df.Volume_y.tolist()
		wday = df.Weekday_x.tolist()

		print("Doing ternary tensor with time length of ",len(Time)," and ", STEP , "steps")

		for i in range(0, len(Time)-self.forecast-self.window, STEP):
			#print (i)

			if(wday[i] != wday[i+self.window]):
				#print("WARNING: We crossed the day boundary for our tensor..skipping")
				continue

			try:
				#ask open, ask high.. bid close, bid volume
				ao = aO[i:i+self.window]
				ah = aH[i:i+self.window]
				al = aL[i:i+self.window]
				ac = aC[i:i+self.window]
				av = aV[i:i+self.window]
            
				if(np.sum(av) == 0 and self.training > 0):
					av[0]=0.001
 
            			#zscore on time window interval
				ao = (np.array(ao) - np.mean(ao)) / np.std(ao)
				ah = (np.array(ah) - np.mean(ah)) / np.std(ah)
				al = (np.array(al) - np.mean(al)) / np.std(al)
				ac = (np.array(ac) - np.mean(ac)) / np.std(ac)

				bo = bO[i:i+self.window]
				bh = bH[i:i+self.window]
				bl = bL[i:i+self.window]
				bc = bC[i:i+self.window]
				bv = bV[i:i+self.window]
				#zscore on time window interval
				bo = (np.array(bo) - np.mean(bo)) / np.std(bo)
				bh = (np.array(bh) - np.mean(bh)) / np.std(bh)
				bl = (np.array(bl) - np.mean(bl)) / np.std(bl)
				bc = (np.array(bc) - np.mean(bc)) / np.std(bc)

				x_i = np.column_stack((ao,ah,al,ac,av,bo,bh,bl,bc,av))

				#no action scenario

				if(self.forecast > 0):
					bet_ask = aC[i+self.window]
					bet_bid = bC[i+self.window]
					bet_time = Time[i+self.window]
					bet_day = wday[i+self.window]

					prediction_ask = aC[i+self.window+self.forecast]
					prediction_bid = bC[i+self.window+self.forecast]
					prediction_time = Time[i+self.window+self.forecast]
					prediction_day = wday[i+self.window+self.forecast]

					if(bet_day != prediction_day):
						#print("WARNING: We crossed the day boundary for our forecast tensor....skipping")
						continue
    
					#bet_time=datetime.datetime.strptime(bet_time, '%d.%m.%Y %H:%M:%S.%f %Z%z')
					if(prediction_bid>bet_ask):
						#if the bid price at prediction time is greater then ask price at bet time: is a put
						y_i = [1,0,0]
					elif(prediction_ask<bet_bid):
						#if the ask price at prediction time is lower then bid price at bet time: is a call
						y_i = [0,1,0]
					else:
						y_i = [0,0,1]

					Y.append(y_i)

				if(count%500 == 0):
					print (count, 'windows sized')
					#print (y_i,bet_ask,bet_bid,bet_time,prediction_ask,prediction_bid,prediction_time)
					#print(ao,ah,al,ac,av,bo,bh,bl,bc,av)
				
				# WARNING: Always do this after the forecast check for next day. Otherwise X may be longer than Y
				X.append(x_i)

			except Exception as e:
				print (e)
				pass

			count=count+1

		if(self.forecast > 0):
			return X,Y
		else:
			return X

	def get_splunk_quotes(self,query):
		lines=[]
		out=subprocess.run([SPLUNK_BIN, 'search',query,"-maxout","0"],stdout=subprocess.PIPE, text=True, bufsize=-1)

		for line in out.stdout.splitlines():
			if line.find('_time') == -1 and line.find('--') == -1:
				lines.append(line.split())

		try:
			df=pd.DataFrame.from_records(lines,coerce_float=True)
		except Exception as e:
			print("Dataframe.from_records exception: ",e)
			exit()

		#print(df.head())
		#print("Tail:")
		#print(df.tail())
    

    
		df = df.rename(columns={ df.columns[0] : 'Datetime', df.columns[1] : 'Open', df.columns[2] : 'High', df.columns[3] : 'Low', df.columns[4]: 'Close', df.columns[5] : 'Volume',df.columns[6]: 'Weekday', df.columns[7]: 'Hour', df.columns[8]:'Minute',df.columns[9]:'Second'})
		df[["Open","High","Low","Close","Volume"]] = df[["Open","High","Low","Close","Volume"]].astype(float)
		df[["Minute","Hour","Second"]]=df[["Minute","Hour","Second"]].astype(int)

		return df

