import numpy as np
import keras
import librosa
from librosa.feature import mfcc
from keras.preprocessing.text import Tokenizer
import model_utils
from keras_adversarial import gan_targets

DATA_FILE_DIR="data/"

class SpeechDataGenerator(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, labels,word_index,vocab_size,batch_size=20,sequence_len=20,n_mfcc=100,audio_dim=150,model_type="compressed",shuffle=True):
		'Initialization'
		self.batch_size = batch_size
		self.labels = labels
		self.list_IDs = list_IDs
		self.word_index = word_index
		self.vocab_size = vocab_size
		self.sequence_len = sequence_len
		self.n_mfcc = n_mfcc
		self.audio_dim = audio_dim
		self.model_type=model_type
		self.shuffle = shuffle
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

		# Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

		# Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)
    
	
	def __data_generation(self, list_IDs_temp):
		'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
		# Initialization
		X = np.empty((self.batch_size,self.n_mfcc,self.audio_dim))
		y = np.empty((self.batch_size,self.sequence_len,self.vocab_size), dtype='int') if self.model_type == "compressed" else np.empty((self.batch_size,1,self.sequence_len*self.vocab_size), dtype='int')
		
		# Generate data
		for i, ID in enumerate(list_IDs_temp):
			# Store sample
			X[i,] = model_utils.convert_audio_to_waveform(ID,self.n_mfcc,self.audio_dim)

			# Store class
			if self.model_type == "compressed":
				y[i,] = model_utils.convert_text_to_sequences(self.labels[i],self.word_index,self.vocab_size,self.sequence_len)
			else:
				y[i,0,] = model_utils.convert_text_to_longsequence(self.labels[i],self.word_index,self.vocab_size,self.sequence_len)
			#y.append(self.labels[i])

		return X, y
		
#================================================================================================================================
        
class VideoDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs=None, labels=None, batch_size=32,img_row=256,img_col=256,frame_count=50,n_classes=6,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.frame_count = frame_count
        self.img_row = img_row
        self.img_col = img_col
        self.shuffle = shuffle
        self.n_classes = n_classes
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,1,self.frame_count,self.img_row,self.img_col))
        y = np.empty((self.batch_size,self.n_classes), dtype='int')
		
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,0,] = model_utils.convert_video_to_frames(ID,self.img_row,self.img_col,self.frame_count)

            # Store class
            y[i] = self.labels[i]
            #y.append(self.labels[i])

        return X, y
        
#================================================================================================================================

class ImageDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs=None,batch_size=32,img_row=128,img_col=128,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.img_row = img_row
        self.img_col = img_col
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

    def __data_generation(self, list_IDs_temp):   	
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,self.img_row,self.img_col))
   
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = model_utils.convert_image_to_array(ID,self.img_row,self.img_col)

        return X, gan_targets(X.shape[0])


class CifarImageDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs=None,batch_size=32,img_row=32,img_col=32,shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.img_row = img_row
        self.img_col = img_col
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)
    

    def __data_generation(self, list_IDs_temp):   	
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size,self.img_row,self.img_col),dtype="int")
   
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            X[i,] = ID

        return X, gan_targets(X.shape[0])


if __name__=="__main__":
	filename = "cv-other-dev/sample-000001.mp3"
	wave_mfcc = model_utils.convert_audio_to_waveform(DATA_FILE_DIR+filename,100)
	print(str(wave_mfcc.shape[0]))
	#print(str(wave_mfcc))
	
