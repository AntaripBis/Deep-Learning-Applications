import librosa
from librosa.feature import mfcc
from keras.preprocessing.text import text_to_word_sequence
import math
import numpy as np
import os
import shutil
import model_utils
import json

DELIMITER=","

class AudioAnalyzer:

	def __init__(self):
		self.data_dir = "data"
		self.metadata_file = "cv-other-dev.csv"
		self.mfcc_npy_dir = "mfcc_npy_dir"
		self.label_file = "test_label.txt"
		self.instances_count = 10
		self.frames_per_word = 20
		self.mfcc_dim = 20
		self.index_json = "word_index.json"
		self.audio_analysis_flag = False
		if self.audio_analysis_flag:
			self.analyze_audio_data()
		else:
			self.generate_word_mfcc()
		
	def analyze_audio_data(self):
		X,Y = self.load_metadata()
		frame_per_char_list = []
		frame_per_word_list = []
		max_frame_per_char = 0
		max_frame_per_word = 0
		for i,audio_file in enumerate(X):
			print("%d-th instance" % (i+1))
			wave_mfcc = model_utils.convert_audio_to_basewave(audio_file)
			#print("Y : %s" % (Y[i]))
			y_len = len(Y[i].replace(" ",""))
			#print("string length : %d count of audio frames : %d mfcc coefficients : %d" % (y_len,wave_mfcc.shape[1],wave_mfcc.shape[0]))
			frame_per_char = int(math.ceil(wave_mfcc.shape[1]/y_len))
			frame_per_char_list.append(frame_per_char)
			if frame_per_char > max_frame_per_char:
				max_frame_per_char = frame_per_char
			text_tokens = text_to_word_sequence(Y[i])
			for token in text_tokens:
				frame_per_word = frame_per_char * len(token)
				frame_per_word_list.append(frame_per_word)
				if frame_per_word > max_frame_per_word:
					max_frame_per_word = frame_per_word
		frame_per_char_list = np.array(frame_per_char_list)
		frame_per_word_list = np.array(frame_per_word_list)
		print("Max frame per character : %d" % (max_frame_per_char))
		print("Max frame per word : %d" % (max_frame_per_word))
		print("Mean frame per character : %f" % np.mean(frame_per_char_list))
		print("Std. deviation of  frame per word : %f" % np.std(frame_per_char_list))
		print("Mean frame per word : %f" % np.mean(frame_per_word_list))
		print("Std. deviation of  frame per word : %f" % np.std(frame_per_word_list))
		
	def convert_text_to_word_mfcc(self,filename,label):
		token_mfcc_list = []
		wave_mfcc = model_utils.convert_audio_to_basewave(filename)
		total_frames = wave_mfcc.shape[1]
		text_tokens = text_to_word_sequence(label)
		start_frame_idx = 0
		for token in text_tokens:
			token_mfcc = np.zeros((self.mfcc_dim,self.frames_per_word))
			seq_len = total_frames - start_frame_idx - 1
			if seq_len >= self.frames_per_word: 
				token_mfcc = wave_mfcc[:,start_frame_idx:start_frame_idx+self.frames_per_word]
			elif seq_len > 0 and seq_len < self.frames_per_word:
				token_mfcc[:,0:seq_len] = wave_mfcc[:,start_frame_idx:start_frame_idx+seq_len]
			start_frame_idx += self.frames_per_word
			token_mfcc_list.append(token_mfcc)
		return token_mfcc_list,text_tokens
		
	def generate_word_mfcc(self):
		X,Y = self.load_metadata()
		total_mfcc_list = []
		total_token_list = []
		valid_token_list = []
		new_path = "/".join([self.data_dir,self.mfcc_npy_dir])
		if os.path.exists(new_path):
			try:
				shutil.rmtree(new_path)
			except e:
				print("Error: %s - %s." % (e.filename, e.strerror))
		print(("="*10)+"Existing directories deleted"+("="*10))
		os.makedirs("/".join([new_path,"X_data"]))
		os.makedirs("/".join([new_path,"Y_data"]))
		print(("="*10)+"New_directories created"+("="*10))
		counter = 0
		for i,label in enumerate(Y):
			#if i > 10:
			#	break
			print("%d - th data point analyzed" % (i+1))
			token_mfcc_list,label_tokens = self.convert_text_to_word_mfcc(X[i],label)
			for j in range(len(label_tokens)):
				if np.max(token_mfcc_list[j]) < 0.01:
					continue
				np.save("/".join([new_path,"X_data","%d_%s.npy" % (counter,label_tokens[j])]),token_mfcc_list[j])
				valid_token_list.append(label_tokens[j])
				counter += 1	
		print(("="*10)+"MFCC data saved to directory"+("="*10))
		#print("Valid Token List length : %d" % (len(valid_token_list)))
		word_index,word_index_len = model_utils.generate_word_index(np.array(valid_token_list))
		for i,token in enumerate(valid_token_list):
			np.save("/".join([new_path,"Y_data","%d_%s.npy" % (i,token)]),model_utils.convert_word_to_sequence(token,word_index,word_index_len))
		print(("="*10)+"Label data saved to directory"+("="*10))
		word_index_path = "/".join([new_path,self.index_json])
		if os.path.exists(word_index_path):
			os.remove(word_index_path)
		with open(word_index_path,"w") as writer:
			writer.write(json.dumps(word_index))
		print(("="*10)+"Word Index saved to file"+("="*10))
								
		
	def load_metadata(self):
		X = []
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
		
    	
if __name__=="__main__":
	AudioAnalyzer()

