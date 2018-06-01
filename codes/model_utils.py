import librosa
from librosa.feature import mfcc
import numpy as np
from cv2 import VideoCapture,resize, cvtColor,destroyAllWindows
import cv2
from scipy.misc import imread,imresize

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
import keras.backend as K

def convert_text_to_sequences(text,word_index,vocab_size,sequence_len=30):
	encoded_text = np.zeros((sequence_len,vocab_size))
	text_words = text_to_word_sequence(text)
	#print(text_words)
	for i,word in enumerate(text_words):
		if i < sequence_len:
			encoded_text[i][word_index[word]-1] = 1 
			#print("Word index : %d "
		#print("Sum : %d" % (np.sum(encoded_text[i])))
	return encoded_text
	
def convert_word_to_sequence(word,word_index,sequence_len):
	encoded_text = np.zeros((sequence_len))
	encoded_text[word_index[word]-1] = 1 
	return encoded_text
			
def convert_text_to_longsequence(text,word_index,vocab_size,sequence_len=30):
	encoded_text = np.zeros(sequence_len*vocab_size)
	text_words = text_to_word_sequence(text)
	#print(text_words)
	for i,word in enumerate(text_words):
		encoded_text[i*sequence_len+word_index[word]-1] = 1 
	return encoded_text

def generate_word_index(texts):
	t = Tokenizer()
	t.fit_on_texts(texts)
	return t.word_index,len(t.word_index.keys())
		
'''    
def convert_audio_to_waveform(filename,mfcc_dim):
    data,sampling_rate = librosa.load(filename)
    wave_mfcc = np.mean(mfcc(y=data, sr=sampling_rate, n_mfcc=mfcc_dim).T,axis=0)
    return wave_mfcc
'''

    
def convert_audio_to_waveform(filename,mfcc_dim=20,audio_dim=150):
	#print("audio file : "+filename)
	data,sampling_rate = librosa.load(filename)
	wave_mfcc = mfcc(y=data, sr=sampling_rate, n_mfcc=mfcc_dim)
	#wave_mfcc = wave_mfcc.resize((mfcc_dim,100))
	audio_form = np.zeros((mfcc_dim,audio_dim))
	for i in range(mfcc_dim):
		 if len(wave_mfcc[i,]) >= audio_dim:
		 	audio_form[i,] = wave_mfcc[i,:audio_dim]
		 else:
		 	audio_form[i,:len(wave_mfcc[i,])] = wave_mfcc[i,] 
	return audio_form.reshape((audio_dim,mfcc_dim))

def convert_video_to_frames(filename,image_rows,image_cols,frame_count=50):
	frames = []
	cap = VideoCapture(filename)
	'''
	fps = cap.get(cv2.CAP_PROP_FPS)
	frame_count_1 = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
	print("frame count : %d" %(frame_count_1))
	width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
	print("frame width : %d" %(width))
	height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
	print("frame height : %d" %(height))
	'''
	for i in range(frame_count):
		ret,frame = cap.read()
		gray = cvtColor(frame, cv2.COLOR_BGR2GRAY)
		#print("Initial frame shape : %s" % str(gray.shape))
		gray = resize(gray,(image_cols,image_rows), interpolation = cv2.INTER_AREA)
		frames.append(gray)
		if cv2.waitKey(10) == 27:
			break
	cap.release()
	destroyAllWindows()
	ipt = np.array(frames)
	'''
	print(ipt.shape)
	print(str(ipt))
	ipt = np.rollaxis(np.rollaxis(ipt,2,0),2,0)
	print("After rollaxis : "+str(ipt))
	'''
	return ipt

def convert_images_to_array(filename,img_row=128,img_col=128):
	img = imread(filename,flatten = True)
	img = img.astype('float32')
	#print("Image before shaping"+str(img.shape))
	img = resize(img,(img_row,img_col))
	return img
	
def convert_audio_to_basewave(filename):
	data,sampling_rate = librosa.load(filename)
	wave_mfcc = mfcc(y=data, sr=sampling_rate)
	return wave_mfcc
	
def convert_speech_to_mfcc(filename,mfcc_dim,frames_per_word):
	token_mfcc_list = []
	#print("MFCC file up for conversion : %s" % (filename))
	wave_mfcc = convert_audio_to_basewave(filename)
	total_frames = wave_mfcc.shape[1]
	start_frame_idx = 0
	while start_frame_idx < total_frames:
		token_mfcc = np.zeros((mfcc_dim,frames_per_word))
		seq_len = total_frames-1 - start_frame_idx
		if seq_len >= frames_per_word: 
			token_mfcc = wave_mfcc[:,start_frame_idx:start_frame_idx+frames_per_word]
		elif seq_len > 0 and seq_len < frames_per_word:
			token_mfcc[:,0:seq_len] = wave_mfcc[:,start_frame_idx:start_frame_idx+seq_len]
		start_frame_idx += frames_per_word
		token_mfcc_list.append(token_mfcc)
	return token_mfcc_list

if __name__=="__main__":
	audio_file = "data/cv-other-dev/sample-000000.mp3"
	wave_mfcc = convert_audio_to_waveform(audio_file)
	print(wave_mfcc.shape)

	'''	
	video_file = "data/kth_videos/boxing/person02_boxing_d1_uncomp.avi"
	frames = convert_video_to_frames(video_file,120,160, 300)
	print(frames.shape)
	
	image_file = "data/images_data/the-simpsons-characters-dataset/simpsons_dataset/agnes_skinner/pic_0000.jpg"
	img_array = convert_images_to_array(image_file)
	print(img_array.shape)
	'''
	
	'''
	text_list = ["They are the worst","We live for being the legend","We rock and we will continue to be the best rockers"]
	word_index,vocab_size,encoded_texts = convert_text_to_sequences(text_list,30)
	print(encoded_texts)
	'''
		
	
