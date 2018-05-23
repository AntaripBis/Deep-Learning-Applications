import librosa
from librosa.feature import mfcc
from keras.preprocessing.text import text_to_word_sequence
import math

DELIMITER=","

class AudioAnalyzer:

	def __init__(self):
		self.data_dir = "data"
		self.metadata_file = "cv-other-dev.csv"
		self.instances_count = 10
		
	def analyze_audio_data(self):
		X,Y = self.load_metadata()
		max_frame_per_char = 0
		max_frame_per_word = 0
		for i,audio_file in enumerate(X):
			print("%d-th instance" % (i+1))
			wave_mfcc = self.convert_audio_to_waveform(audio_file)
			#print("Y : %s" % (Y[i]))
			y_len = len(Y[i].replace(" ",""))
			#print("string length : %d count of audio frames : %d mfcc coefficients : %d" % (y_len,wave_mfcc.shape[1],wave_mfcc.shape[0]))
			frame_per_char = int(math.ceil(wave_mfcc.shape[1]/y_len))
			if frame_per_char > max_frame_per_char:
				max_frame_per_char = frame_per_char
			text_tokens = text_to_word_sequence(Y[i])
			for token in text_tokens:
				frame_per_word = frame_per_char * len(token)
				if frame_per_word > max_frame_per_word:
					max_frame_per_word = frame_per_word
		print("Max frame per character : %d" % (max_frame_per_char))
		print("Max frame per word : %d" % (max_frame_per_word))
			
		
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
		
	def convert_audio_to_waveform(self,filename):
		data,sampling_rate = librosa.load(filename)
		wave_mfcc = mfcc(y=data, sr=sampling_rate)
		return wave_mfcc
    	
if __name__=="__main__":
	audio_obj = AudioAnalyzer()
	audio_obj.analyze_audio_data()

