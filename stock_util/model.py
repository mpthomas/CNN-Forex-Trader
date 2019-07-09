import pandas as pd
import numpy as np
import random
import matplotlib.pylab as plt
from datetime import *
import time
import locale
locale.setlocale(locale.LC_NUMERIC, "")
import subprocess
import json
import uuid

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Conv1D, MaxPooling1D, AtrousConvolution1D, RepeatVector
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback
from keras.layers.wrappers import Bidirectional
from keras import regularizers
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import *
from keras.optimizers import RMSprop, Adam, SGD, Nadam
from keras.initializers import *
from keras.backend import clear_session

class TrainLogger(Callback):
	def on_epoch_end(self,epoch,logs):
		if(self.model.parent_obj.log_flag == 0):
			return

		self.model.parent_obj.epoch=self.model.parent_obj.epoch+1

		self.model.parent_obj.time=datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")
		if np.isnan(logs['loss']):
			logs['loss'] = 0
		if np.isnan(logs['val_loss']):
			logs['val_loss'] = 0
		if np.isnan(logs['acc']):
			logs['acc'] = 0
		if np.isnan(logs['val_acc']):
			logs['val_acc'] = 0

		#self.parent.loggerfp.write(
		self.model.parent_obj.loggerfp.write(json.dumps({'time': self.model.parent_obj.time,
	  		'id': self.model.parent_obj.uuid,
	  		'loss' : logs['loss'],
	  		'val_loss' : logs['val_loss'],
	  		'acc' : logs['acc'],
	  		'val_acc' : logs['val_acc'],
	  		'window': self.model.parent_obj.window,
	  		'forecast' : self.model.parent_obj.forecast,
	  		'epochs' : self.model.parent_obj.epochs,
	  		'epoch' : self.model.parent_obj.epoch,
	  		'load_file' : self.model.parent_obj.load_file,
	  		'train_len' : self.model.parent_obj.train_len,
	  		'val_len' : self.model.parent_obj.val_len,
	  		'batch_size' : self.model.parent_obj.batch_size,
	  		'patience' : self.model.parent_obj.patience,
	  		'lr' : self.model.parent_obj.lr,
	  		'lr_reduce_factor' : self.model.parent_obj.lr_reduce_factor,
	  		'test_name' : self.model.parent_obj.test_name,
	  		'test_desc' : self.model.parent_obj.test_desc,
	  		'save_file' : self.model.parent_obj.save_file})+'\n')
		#print("Inside _epoch_end")

	def on_train_begin(self,logs):
		self.model.parent_obj.uuid=str(uuid.uuid1())
		self.model.parent_obj.loggerfp=open('/Users/matt/xcode/stock/train_log.log',mode='a+t', buffering=1)
		self.model.parent_obj.epoch=0

class Model():

	def __init__(self,height,width,load_file,save_file,forecast,epochs=10):
		self.time=datetime.now().strftime("%m-%d-%Y %H:%M:%S.%f")
		self.height=height
		self.width=width
		self.window=height
		self.load_file=load_file
		self.save_file=save_file
		self.forecast=forecast
		self.epochs=epochs
		self.epoch=0
		self.uuid=''
		self.train_length=0
		self.val_length=0
		self.test_name="Default Test"
		self.test_desc="Default Desc"

		# Network hyperparameters
		self.batch_size=86
		self.patience=30
		self.lr=0.002
		self.lr_reduce_factor=0.9
		
		self.set_logging(0)

	def set_logging(self,flag):
		self.log_flag=flag
	
	def reset_model(self):
		clear_session()
		self.model=self.build_model()

	def build_model(self):
		model=Sequential()

		model.add(Conv1D(input_shape=(self.height,self.width), filters=16, kernel_size=4, padding="same"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.5))

		model.add(Conv1D(filters=8, kernel_size=4, padding="same"))
		model.add(BatchNormalization())
		model.add(LeakyReLU())
		model.add(Dropout(0.5))

		model.add(Flatten())

		model.add(Dense(64))
		model.add(BatchNormalization())
		model.add(LeakyReLU())

		model.add(Dense(3))
		model.add(Activation('softmax'))


		opt = Nadam(lr=self.lr)
		self.reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=self.lr_reduce_factor, patience=self.patience, min_lr=0.000001, verbose=1)
		self.checkpointer = ModelCheckpoint(filepath=self.save_file, monitor='val_loss',verbose=1, save_best_only=True,mode='auto')
		model.parent_obj=self

		self.logger=TrainLogger()

		#self.logger = LambdaCallback(
		#	on_train_begin=lambda logs: self._train_begin,
		#	on_epoch_end=lambda epoch, logs: print("Forecast: ",model.parent.forecast),
		#	on_train_end=lambda logs: self._train_end
		#	)

		model.compile(optimizer=opt, 
				loss='categorical_crossentropy',
				metrics=['accuracy'])

		self.model=model

		return model

	def _train_end(self,logs):
		self.loggerfp.close()

	def train(self, X_train, y_train, X_test, y_test,log=0):
		self.train_len=len(X_train)
		self.val_len=len(X_test)

		history=self.model.fit(X_train, y_train,
				epochs = self.epochs,
				batch_size = self.batch_size, 
				verbose=1,
				validation_data=(X_test, y_test),
				callbacks=[self.reduce_lr, self.checkpointer, self.logger],
				shuffle=True)
	
	def predict(self, X_test):
		self.model.load_weights(self.save_file)
		pred=self.model.predict(X_test)

		return pred
