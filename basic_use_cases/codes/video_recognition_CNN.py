from os import listdir
from os.path import join, isfile
import os

import numpy as np
import math
from random import sample

from keras.layers.convolutional import Conv3D,MaxPooling3D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model,Sequential,model_from_json,load_model
from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint
from keras.regularizers import L1L2
import keras.utils

from input_data import VideoDataGenerator
import model_utils

class ActionRecognizer:
	
	def __init__(self):
		self.reg = lambda : L1L2(1e-05,1e-05)
		self.read_configuration("config/video_recognition_CNN.prop")
		#print("Video Dimension : (%d,%d,%d)" % (self.frame_count,self.img_rows,self.img_cols))
		self.run_video_recognition()
		
	def read_configuration(self,property_file):
		with open(property_file,"r") as config_reader:
			lines = [line.rstrip("\n").split("=") for line in config_reader.readlines() if not line.startswith("#")]
			key_value_pairs = [(pair[0],pair[1]) for pair in lines]
			lines = None
			for (key,value) in key_value_pairs:
				if key == "DATA_DIR":
					self.data_dir = value
				elif key == "MODEL_DIR":
					self.model_dir = value
				elif key == "RESULT_DIR":
					self.result_dir = value
				elif key == "MODEL_JSON_FILE":
					self.model_json_file = value
				elif key == "MODEL_H5_FILE":
					self.model_h5_file = value
				elif key == "RESULT_FILE":
					self.result_file = value
				elif key == "BATCH_SIZE":
					self.batch_size = int(value)
				elif key == "FRAME_COUNT":
					self.frame_count = int(value)
				elif key == "TRAINING_RATIO":
					self.training_ratio = float(value)
				elif key == "FILTERS":
					layer_values = value.split(",")
					self.filters = [int(layer_value) for layer_value in layer_values]
				elif key == "POOL":
					layer_values = value.split(",")
					self.pool = [int(layer_value) for layer_value in layer_values]
				elif key == "CONVOLUTION_LAYERS":
					layer_values = value.split(",")
					self.convolution_layers = [int(layer_value) for layer_value in layer_values]
				elif key == "IMAGE_ROWS":
					self.img_rows = int(value)
				elif key == "IMAGE_COLS":
					self.img_cols = int(value)
				elif key == "DROPOUT_RATE":
					self.dropout_rate = float(value)
				elif key == "CONNECTED_LAYERS":
					layer_values = value.split(",")
					self.connected_layers = [int(layer_value) for layer_value in layer_values]
				elif key == "LOAD_PREVIOUS_MODEL_FLAG":
					self.load_previous_model = True if value == "true" else False
				elif key == "EPOCHS":
					self.epochs = int(value)
				elif key == "NUM_CHANNELS":
					self.n_channels = int(value)
				elif key == "LOSS_FUNC":
					self.loss_fn = value
				elif key == "METRICS":
					self.metrics = value.split(",")
				elif key == "CATEGORIES":
					self.category_dict = {category:i for i,category in enumerate(value.split(","))}
					self.reverse_category_dict = {value:key for (key,value) in self.category_dict.items()}
					self.n_classes = len(self.category_dict.keys())
				elif key == "CLASSIFY_VIDEO_FLAG":
					self.classify_video_flag = True if value == "true" else False
				elif key == "LEARNING_RATE":
					self.learning_rate = float(value)
			key_value_pairs = None
			config_reader.close()
				
		
	def run_video_recognition(self):
		X,Y = self.load_metadata()
		print(("-"*20)+"Metadata loaded"+("-"*20))
		Y = keras.utils.to_categorical(Y,num_classes = self.n_classes)
		(x_train,y_train),(x_validation,y_validation) = self.partition_data(X,Y)
		X = Y = None
		
		training_generator = VideoDataGenerator(x_train,y_train,self.batch_size,self.img_rows,self.img_cols, self.frame_count,self.n_classes,self.n_channels)
		print(("-"*20)+"Prepared generator for training data"+("-"*20))
		validation_generator = VideoDataGenerator(x_validation,y_validation,self.batch_size,self.img_rows,self.img_cols, self.frame_count,self.n_classes,self.n_channels)
		print(("-"*20)+"Prepared generator for validation data"+("-"*20))
		
		model = None
		if self.load_previous_model and os.path.exists("/".join([self.model_dir,self.model_json_file])):
			# load json and create model
			json_file = open("/".join([self.model_dir,self.model_json_file]), 'r')
			loaded_model_json = json_file.read()
			json_file.close()
			model = model_from_json(loaded_model_json)
			# load weights into new model
			model.load_weights("/".join([self.model_dir,self.model_h5_file]))
			model = self.compile_model(model)
			print(("="*10)+"Loaded model from disk"+("="*10))
		else:
			model = self.build_model()
			print(("-"*20)+"Model Built with required parameters"+("-"*20))
			model = self.train_model(model, training_generator, validation_generator)
			print(("-"*20)+"Model training completed"+("-"*20))
		
		if self.classify_video_flag:
			self.classify_videos(model,x_validation,y_validation)
		
	def partition_data(self,X,Y):
		#print("X : "+str(X.shape))
		#print("Y : "+str(Y.shape))
		train_data_count = int(math.floor(X.shape[0]*self.training_ratio))
		print("Training examples count : %d" % (train_data_count))
		index = sample(range(X.shape[0]),train_data_count)
		#print(index.shape)
		x_train = X[index]
		y_train = Y[index]
		x_validation = [X[i] for i in range(len(X)) if i not in index]
		y_validation = [Y[i] for i in range(len(Y)) if i not in index]
		'''
		x_validation = np.delete(X,index)
		y_validation = np.delete(Y,index)
		
		for i,x in enumerate(x_validation):
			print("X : %s Y : %s" % (x,str(y_validation[i])))
		'''
		return (x_train,y_train),(x_validation,y_validation)	
		
	def compile_model(self,model):
		#Build optimizer
		model_optimizer = RMSprop(lr=self.learning_rate)
		
		#Compile models with the training parameters
		model.compile(loss=self.loss_fn,optimizer=model_optimizer,metrics=self.metrics)
		
		return model
	
	def train_model(self,model,training_generator,validation_generator=None):
		if validation_generator is None:
			validation_generator = training_generator
		
		model = self.compile_model(model)
		
		#train the model
		model.fit_generator(generator=training_generator, validation_data = validation_generator,epochs = self.epochs,use_multiprocessing=True,workers=2)
		
		if not self.load_previous_model:
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir)
			
			# serialize model to JSON
			model_json = model.to_json()
			with open("/".join([self.model_dir,self.model_json_file]), "w") as json_file:
				json_file.write(model_json)
				json_file.close()
			# serialize weights to HDF5
			model.save_weights("/".join([self.model_dir,self.model_h5_file]))

			print(("="*10)+"Saved model to disk"+("="*10))
	
	def load_metadata(self):
		X = []
		Y = []
		action_dirs = listdir(self.data_dir)
		for action_dir in action_dirs:
			if not isfile(action_dir):
				new_path = "/".join([self.data_dir,action_dir])
				action_files = listdir(new_path)
				for action_file in action_files:
					X.append("/".join([new_path,action_file]))
					Y.append(self.category_dict[action_dir])
					#print("X : %s Y : %d" % (X[len(X)-1],Y[len(Y)-1]))
		#print(self.category_dict)
		return np.array(X),np.array(Y)
		
	def build_model(self):
		model = Sequential()
		if len(self.convolution_layers) > 0 and len(self.pool) > 0:
			#print(self.convolution_layers[0])
			model.add(Conv3D(self.filters[0],kernel_size=self.convolution_layers[0],input_shape=(self.n_channels,self.frame_count,self.img_rows,self.img_cols),activation="relu",data_format="channels_first",kernel_regularizer = self.reg(),bias_regularizer = self.reg()))
			model.add(MaxPooling3D(pool_size=(self.pool[0],self.pool[0],self.pool[0])))
			model.add(Dropout(self.dropout_rate))
			if len(self.convolution_layers) >= 2:
				for i in range(1,len(self.convolution_layers)):
					model.add(Conv3D(self.filters[i],kernel_size=self.convolution_layers[i],activation="relu",data_format="channels_first",kernel_regularizer = self.reg(),bias_regularizer = self.reg()))
					model.add(MaxPooling3D(pool_size=(self.pool[i],self.pool[i],self.pool[i])))
					model.add(Dropout(self.dropout_rate))
		'Get the final dense layer'
		model.add(Flatten())
		if len(self.connected_layers) > 0:
			for i in range(len(self.connected_layers)):
				model.add(Dense(self.connected_layers[i],activation="relu",kernel_regularizer = self.reg(),bias_regularizer = self.reg()))
		model.add(Dense(self.n_classes,kernel_initializer='normal',kernel_regularizer = self.reg(),bias_regularizer = self.reg()))
		model.add(Activation('softmax'))
		print(model.summary())
		return model
		
	def classify_videos(self,model=None,video_files = None,label_texts=None):
		if model is None or video_files is None:
			return
		if not os.path.exists(self.result_dir):
			os.makedirs(self.result_dir)
		#print("reverse category dict : %s" % (str(self.reverse_category_dict)))
		with open("/".join([self.result_dir,self.result_file]),"w") as writer:
			for i,video_file in enumerate(video_files):
				#if i > 30:
				#	break
				predicted_label = self.predict_label_for_video(model,video_file)
				#print("predicted label : %s" % (predicted_label))
				largest_idx = np.where(label_texts[i] == np.max(label_texts[i]))
				largest_idx = largest_idx[0][0]
				#print("Label Texts : %s" % (self.reverse_category_dict[largest_idx]))
				writer.write("|\t|".join([video_file,self.reverse_category_dict[largest_idx],predicted_label])+"\n")
				print("%d - th classification completed" % (i+1))
			writer.close()
			
			
		
	def predict_label_for_video(self,model,video_file):
		video_data = model_utils.convert_video_to_frames(video_file,self.img_rows,self.img_cols,self.frame_count,self.n_channels)
		#print(video_data.shape)
		video_data = video_data.reshape((1,self.n_channels,self.frame_count,self.img_rows,self.img_cols))
		label_list =  model.predict_on_batch(video_data)
		#print(label_list)
		max_idx = 0
		max_val = label_list[0,0]
		for i in range(1,label_list.shape[1]):
			if max_val < label_list[0,i]:
				max_val = label_list[0,i]
				max_idx = i
		#print("Maximum ID : %d" % (max_idx))
		return self.reverse_category_dict[max_idx]

if __name__=="__main__":
	ActionRecognizer()
	'''
	X,y = recognizer_obj.load_metadata()
	print("Count of entries : "+str(len(X)))
	print(X[0:5])
	print(y[0:5])
	'''
				

