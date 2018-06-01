"""Model definitions for simple speech recognition.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import math
import sys
import numpy as np
from random import sample
import json

from keras.optimizers import RMSprop,Adam
from keras.callbacks import ModelCheckpoint
from keras.layers import Input,BatchNormalization,TimeDistributed,Dense,Activation,LSTM,Reshape,RepeatVector,Dropout
from keras.models import Model,Sequential,model_from_json,load_model
from keras.losses import categorical_crossentropy
from keras.regularizers import L1L2

from pathlib import Path
from scipy import spatial


from input_data import SpeechDataGenerator,SpeechWordDataGenerator
from audio_analysis import AudioAnalyzer
import model_utils

DELIMITER=","
		
class SpeechTextTranslator:		
		
	def __init__(self):
		self.reg = lambda: L1L2(1e-5, 1e-5)
		self.read_configuration("config/speech_translation_RNN.prop")
		#print("metadata_file : %s" % (self.metadata_file))
		self.run_speech_translation()
		
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
				elif key == "MFCC_DIR":
					self.mfcc_dir = value
				elif key == "MODEL_JSON_FILE":
					self.model_json_file = value
				elif key == "MODEL_H5_FILE":
					self.model_h5_file = value
				elif key == "RESULT_FILE":
					self.result_file = value
				elif key == "METADATA_FILE":
					self.metadata_file = value
				elif key == "ANALYSIS_LEVEL":
					self.analysis_level = value
				elif key == "STACKED_LAYERS":
					layer_values = value.split(",")
					self.stacked_layers = [int(layer_value) for layer_value in layer_values]
				elif key == "ENCODER_LAYERS":
					layer_values = value.split(",")
					self.encoder_layers = [int(layer_value) for layer_value in layer_values]
				elif key == "DECODER_LAYERS":
					layer_values = value.split(",")
					self.decoder_layers = [int(layer_value) for layer_value in layer_values]
				elif key == "BATCH_SIZE":
					self.batch_size = int(value)
				elif key == "TRAINING_RATIO":
					self.training_ratio = float(value)
				elif key == "SEQUENCE_LENGTH":
					self.sequence_len = int(value)
				elif key == "MOMENTUM":
					self.momentum = float(value)
				elif key == "LOAD_PREVIOUS_MODEL_FLAG":
					self.load_previous_model = True if value == "true" else False
				elif key == "EPOCHS":
					self.epochs = int(value)
				elif key == "MODEL_TYPE":
					self.model_type = value
					self.loss_fn = "categorical_crossentropy" if self.model_type ==  "compressed" else "cosine_proximity"
					self.metrics = ["accuracy"] if self.model_type == "compressed" else ["accuracy","cosine"]
				elif key == "N_MFCC":
					self.n_mfcc = int(value)
				elif key == "AUDIO_DIM":
					self.audio_dim = int(value)
				elif key == "PROBABILITY_THRESHOLD":
					self.prob_threshold = float(value)
				elif key == "TRANSLATE_SPEECH_FLAG":
					self.translate_speech_flag = True if value == "true" else False
				elif key == "LEARNING_RATE":
					self.learning_rate = float(value)
				elif key == "DROPOUT_RATE":
					self.dropout_rate = float(value)
			key_value_pairs = None
			config_reader.close()
		
	def run_speech_translation(self):
		self.word_index = None
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
			X,Y = self.load_metadata()
			print(("="*10)+"Metadata loaded"+("="*10))
		
			if self.analysis_level == "sentence":
				self.word_index,self.vocab_size = model_utils.generate_word_index(Y)
		
			self.reverse_word_index = {(value-1):key for (key,value) in self.word_index.items()}
			(x_train,y_train),(x_validation,y_validation) = self.partition_data(X,Y)
			X = Y = None
			print(("="*10)+"Training and Validation data prepared"+("="*10))
		
			training_generator = SpeechDataGenerator(x_train,y_train,self.word_index,self.vocab_size,self.batch_size,self.sequence_len,self.n_mfcc,self.audio_dim,self.model_type) if self.analysis_level == "sentence" else SpeechWordDataGenerator(x_train,y_train,self.vocab_size,self.batch_size,self.n_mfcc,self.audio_dim)
			validation_generator = SpeechDataGenerator(x_validation,y_validation,self.word_index,self.vocab_size,self.batch_size,self.sequence_len,self.n_mfcc,self.audio_dim,self.model_type) if self.analysis_level == "sentence" else SpeechWordDataGenerator(x_validation,y_validation,self.vocab_size,self.batch_size,self.n_mfcc,self.audio_dim)
			
			print(("="*10)+"Training and Validation generators ready"+("="*10))
			
			model = self.build_encoder_decoder_model()
			print(("="*10)+"Model building completed"+("="*10))
			model = self.train_model(model,training_generator,validation_generator)
			print(("="*10)+"Training model completed"+("="*10))
	
		if self.translate_speech_flag:
			x_test,y_test = self.get_audio_metadata()
			if self.word_index is None:
				if self.analysis_level == "sentence":
					self.word_index,self.vocab_size = model_utils.generate_word_index(y_test)
				else:
					current_data_dir = "/".join([self.data_dir,self.mfcc_dir])
					files = os.listdir(current_data_dir)
					files.sort()
					#print(files[2])
					with open("/".join([current_data_dir,files[2]])) as reader:
						self.word_index = json.load(reader)
					self.vocab_size = len(self.word_index.keys())
				self.reverse_word_index = {(value-1):key for (key,value) in self.word_index.items()}
				print(("="*10)+"Word Index created"+("="*10))
			self.translate_speeches(model,x_test,y_test)
			print(("="*10)+"Translation of speeches completed"+("="*10))
		
		
	def partition_data(self,X,Y):
		print("X : "+str(X.shape))
		#print("Y : "+str(Y.shape))
		train_data_count = int(math.floor(X.shape[0]*self.training_ratio))
		print(train_data_count)
		index = sample(range(X.shape[0]),train_data_count)
		#print(index.shape)
		x_train = X[index]
		y_train = Y[index]
		x_validation = np.delete(X,index)
		y_validation = np.delete(Y,index)
		return (x_train,y_train),(x_validation,y_validation)
		
	def compile_model(self,model):
		#Build optimizer
		model_optimizer = Adam(lr=self.learning_rate)
		
		#Compile models with the training parameters
		model.compile(loss=self.loss_fn,optimizer=model_optimizer,metrics=self.metrics)
		
		return model
		
	def train_model(self,model,training_generator,validation_generator=None):
		if validation_generator is None:
			validation_generator = training_generator
		
		model = self.compile_model(model)
		
		#train the model
		model.fit_generator(generator=training_generator, validation_data = validation_generator, epochs=self.epochs,use_multiprocessing=True,workers=2)
		
		if not self.load_previous_model:
			if not os.path.exists(self.model_dir):
				os.makedirs(self.model_dir)
			
			# serialize model to JSON
			model_json = model.to_json()
			with open("/".join([self.model_dir,self.model_json_file]), "w") as json_file:
				json_file.write(model_json)
			# serialize weights to HDF5
			model.save_weights("/".join([self.model_dir,self.model_h5_file]))
			print(("="*10)+"Saved model to disk"+("="*10))
		
		return model
		
		
	def load_metadata(self):
		X = []
		Y = []
		counter = 1
		if self.analysis_level == "sentence":
			X,Y = self.get_audio_metadata()
		else:
			current_data_dir = "/".join([self.data_dir,self.mfcc_dir])
			files = os.listdir(current_data_dir)
			if files is None or len(files) == 0:
				AudioAnalyzer()
				print(("="*10)+"Transformation of Audio is complete"+("="*10))
				files = os.listdir(current_data_dir)
			files.sort()
			X_dir,Y_dir,word_index_file = files
			X_files = os.listdir("/".join([current_data_dir,X_dir]))
			Y_files = os.listdir("/".join([current_data_dir,Y_dir]))
			with open("/".join([current_data_dir,word_index_file])) as reader:
				self.word_index = json.load(reader)
			self.vocab_size = len(self.word_index.keys())
			#print("Vocab size after loading index : %s" % str(self.vocab_size))
			for i,X_file in enumerate(X_files):
				X.append("/".join([current_data_dir,X_dir,X_file]))
				Y.append("/".join([current_data_dir,Y_dir,Y_files[i]]))
				counter += 1
				if counter % 1000 == 0:
					print(str(counter)+" records have been appended")
					
		return np.array(X),np.array(Y)
		
		
	def build_encoder_decoder_model(self):
		#input_data = Input(shape = (self.n_mfcc,self.audio_dim))
		input_data = Input(shape = (self.audio_dim,self.n_mfcc))
		if self.analysis_level != "sentence":
			if len(self.stacked_layers) > 0:
				layer = LSTM(self.stacked_layers[0], activation="relu",return_sequences = True,kernel_regularizer=self.reg(),bias_regularizer=self.reg())(input_data)
				layer = BatchNormalization(momentum = self.momentum)(layer)
				#layer = Dropout(self.dropout_rate)(layer)
				if len(self.stacked_layers) >= 2:
					for i in range(1,len(self.stacked_layers)):
						return_sequence = False if i == len(self.stacked_layers) - 1 else True
						layer = LSTM(self.stacked_layers[i], activation="relu",return_sequences = return_sequence,kernel_regularizer=self.reg(),bias_regularizer=self.reg())(layer)
						layer = BatchNormalization(momentum = self.momentum)(layer)
						#layer = Dropout(self.dropout_rate)(layer)
				layer = Dense(self.vocab_size, activation="relu",kernel_regularizer=self.reg(),bias_regularizer=self.reg())(layer)
				output_data = Activation('softmax', name='final_activation')(layer)
				model = Model(inputs=input_data,outputs=output_data)
				print(("="*10)+"Check if proper model built"+("="*10))
				print(model.summary())
				return model
		
		else:
			if len(self.encoder_layers) > 0 and len(self.decoder_layers) > 0:
				layer = LSTM(self.encoder_layers[0], activation="relu",kernel_regularizer=self.reg(),bias_regularizer=self.reg())(input_data)
				#layer = BatchNormalization(name="bt_rnn_encoder_1",momentum = self.momentum)(layer)
				if len(self.encoder_layers) >= 2:
					for i in range(1,len(self.encoder_layers)):
						layer = Dense(self.encoder_layers[i], activation="relu",kernel_regularizer=self.reg(),bias_regularizer=self.reg())(layer)
				layer = RepeatVector(self.sequence_len)(layer)
				for i in range(len(self.decoder_layers)):
					layer = LSTM(units=self.decoder_layers[i], return_sequences = True,activation="relu",kernel_regularizer=self.reg(),bias_regularizer=self.reg())(layer)
				layer = TimeDistributed(Dense(units=self.vocab_size,kernel_regularizer=self.reg(),bias_regularizer=self.reg()))(layer)
				output_data = Activation('sigmoid', name='final_activation')(layer)
				if self.model_type == "long":
					output_data = Reshape(target_shape=(1,self.sequence_len*self.vocab_size))(output_data)
				model = Model(inputs=input_data,outputs=output_data)
				print(model.summary())
				return model
		return None 
		
	def translate_speeches(self,model=None,audio_files = None,label_texts=None):
		if model is None or audio_files is None:
			return
		if not os.path.exists(self.result_dir):
			os.makedirs(self.result_dir)
		with open("/".join([self.result_dir,self.result_file]),"w") as writer:
			for i,audio_file in enumerate(audio_files):
				if i > 10:
					break
				translated_text = self.get_sentence_from_speech(model,audio_file) if self.analysis_level == "sentence" else self.get_words_from_speech(model,audio_file)
				writer.write("|\t|".join([audio_file,label_texts[i],translated_text])+"\n")
				print("%d - th translation completed" % (i+1))
			writer.close()
			
	def get_audio_metadata(self,X = None,Y = None):
		if X is None:
			X = [] 
		if Y is None:
			Y = []
		counter = 1
		with open("/".join([self.data_dir,self.metadata_file]),"r") as infile:
			for line in infile:
				if not line.startswith("cv"):
					continue
				line_split = line.split(DELIMITER)
				if len(line_split) < 2:
					continue
				X.append("/".join([self.data_dir,line_split[0]]))
				Y.append(line_split[1])
				counter += 1
				if counter % 1000 == 0:
					print(str(counter)+" records have been appended")
		return X,Y
		
	def get_sentence_from_speech(self,model,audio_file):
		audio_data = model_utils.convert_audio_to_waveform(audio_file,self.n_mfcc,self.audio_dim)
		#audio_data = audio_data.reshape((1,self.n_mfcc,self.audio_dim))
		audio_data = audio_data.reshape((1,self.audio_dim,self.n_mfcc))
		predicted_text = model.predict_on_batch(audio_data)
		
		text_words = []
		if self.model_type == "compressed":
			for i in range(predicted_text.shape[1]):
				#print(predicted_text[0,i,])
				max_idx = 0
				max_val = predicted_text[0,0,0]
				for j in range(1,predicted_text.shape[2]):
					if max_val < predicted_text[0,i,j]:
						max_idx = i
						max_val = predicted_text[0,i,j]
				#print("Maximum index : %d " % (max_idx))
				if max_val >= self.prob_threshold:
					text_words.append(self.reverse_word_index[max_idx])
		else:
			for i in range(self.sequence_len):
				current_sequence = predicted_text[0][0][i*self.vocab_size:(i+1)*self.vocab_size]
				#print(current_sequence)
				max_val = current_sequence[0]
				max_idx = 0
				for j in range(1,self.vocab_size):
					if current_sequence[j] > max_val:
						max_val = current_sequence[j]
						max_idx = j
				if max_val >= self.prob_threshold:
					text_words.append(self.reverse_word_index[max_idx])
		return " ".join(text_words)
		
	def get_words_from_speech(self,model,audio_file):
		token_mfcc_list = model_utils.convert_speech_to_mfcc(audio_file,self.n_mfcc,self.audio_dim)
		text_words = []
		for token_mfcc in token_mfcc_list:
			token_mfcc = token_mfcc.reshape((1,self.n_mfcc,self.audio_dim))
			predicted_text = model.predict(token_mfcc)
			max_idx = 0
			max_val = predicted_text[0,0]
			for j in range(1,predicted_text.shape[1]):
				if max_val < predicted_text[0,j]:
					max_idx = j
					max_val = predicted_text[0,j]
			if max_val >= self.prob_threshold:
				text_words.append(self.reverse_word_index[max_idx])
		return " ".join(text_words)
		
			
if __name__=="__main__":
	SpeechTextTranslator()


