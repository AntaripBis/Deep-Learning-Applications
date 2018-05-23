import matplotlib.pyplot as plt
from PIL import Image
import math
import os
import numpy as np
from cv2 import resize, cvtColor
import cv2

from keras.layers import Reshape, Flatten, Dense, InputLayer,Input,BatchNormalization
#from keras.layers.convolutional import UpSampling2D, MaxPooling2D
from keras.models import Sequential,Model
from keras.optimizers import Adam
#from keras.callbacks import TensorBoard
#from keras_adversarial.image_grid_callback import ImageGridCallback
from keras.datasets import cifar10

import keras.backend as K
from keras.regularizers import L1L2

from input_data import ImageDataGenerator,CifarImageDataGenerator



class ImageSynthesis:
	
	def __init__(self):
		'''
		self.data_dir="data/images_data/the-simpsons-characters-dataset/simpsons_dataset"
		self.model_dir = "model/image_synthesis_GAN"
		self.result_dir = "results/gan_images"
		self.loss_fn = "binary_crossentropy"
		self.generator_layers=[256,256,512]
		self.discriminator_layers=[512,256]
		self.latent_dim = 100
		self.img_rows = 32
		self.img_cols = 32
		self.n_channels = 3
		self.use_channel_in_img = True
		self.learning_rate=0.01
		self.batch_size=200
		self.batch_num = 10
		self.generate_images_flag = False
		self.load_previous_model = False

		self.use_cifar_flag=True
		self.plot_rows=5
		self.plot_cols=5
		'''
		self.read_configuration("config/image_synthesis_GAN.prop")
		self.reg = lambda: L1L2(1e-5, 1e-5)
		self.img_shape = (self.img_rows,self.img_cols,self.n_channels)
		self.run_image_synthesis()
		
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
				elif key == "METADATA_FILE":
					self.metadata_file = value
				elif key == "GENERATOR_LAYERS":
					layer_values = value.split(",")
					self.generator_layers = [int(layer_value) for layer_value in layer_values]
				elif key == "DISCRIMINATOR_LAYERS":
					layer_values = value.split(",")
					self.discriminator_layers = [int(layer_value) for layer_value in layer_values]
				elif key == "BATCH_SIZE":
					self.batch_size = int(value)
				elif key == "LATENT_DIMENSION":
					self.latent_dim = int(value)
				elif key == "GENERATE_BATCH_NUM":
					self.batch_num = int(value)
				elif key == "USE_CIFAR_FLAG":
					self.use_cifar_flag = True if value == "true" else False
				elif key == "LOAD_PREVIOUS_MODEL_FLAG":
					self.load_previous_model = True if value == "true" else False
				elif key == "EPOCHS":
					self.epochs = int(value)
				elif key == "LOSS_FUNCTION":
					self.loss_fn = value
				elif key == "METRICS":
					self.metrics = value.split(",")
				elif key == "IMAGE_ROWS":
					self.img_rows = int(value)
				elif key == "IMAGE_COLUMNS":
					self.img_cols = int(value)
				elif key == "PLOT_ROWS":
					self.plot_rows = int(value)
				elif key == "PLOT_COLUMNS":
					self.plot_cols = int(value)
				elif key == "CHANNEL_NUM":
					self.n_channels = int(value)
				elif key == "GENERATE_IMAGE_FLAG":
					self.generate_images_flag = True if value == "true" else False
				elif key == "LEARNING_RATE":
					self.learning_rate = float(value)
			key_value_pairs = None
			config_reader.close()

		
	def run_image_synthesis(self):
		if self.use_cifar_flag:
			self.x_train = self.load_cifar_data()
			print(("-"*20)+"Cifar data loaded"+("-"*20))
			discriminator,generator,combined_model = self.build_gan()
			print(("-"*20)+"Model building completed"+("-"*20))
			generator = self.train_gan(discriminator,generator,combined_model)
			if self.generate_images_flag:
				self.generate_and_save_images(generator)
		'''
		X = self.load_metadata()
		print(("-"*20)+"Metadata loaded"+("-"*20))
		training_generator = ImageDataGenerator(X,self.batch_size, self.img_rows,self.img_cols)
		print(("-"*20)+"Prepared generator for training data"+("-"*20))
		self.build_and_train_gan(training_generator)
		print(("-"*20)+"Model training completed"+("-"*20))
		'''
		
	def load_metadata(self):
		X = []	
		character_dirs=os.listdir(self.data_dir)
		for character_dir in character_dirs:
			new_path = os.path.join(self.data_dir,character_dir)
			image_files=os.listdir(new_path)
			for image_file in image_files:
				X.append(os.path.join(new_path,image_file))
		return X
		
	def load_cifar_data(self):
		(x_train,y_train),(x_test,y_test) = cifar10.load_data()
		new_train_x = np.array(x_train)
		new_train_x = (new_train_x - 127.5)/127/5 
		return new_train_x
		
	def build_generator(self):
		self.noise_shape = (self.latent_dim,)
		model = Sequential()
		for i in range(len(self.generator_layers)):
			activation_fn = "tanh" if (i == len(self.generator_layers)-1) else "relu"
			if i == 0:
				model.add(Dense(units = self.generator_layers[i], input_shape = self.noise_shape,activation = activation_fn, kernel_regularizer=self.reg()))
			else:
				model.add(Dense(units = self.generator_layers[i],activation = activation_fn, kernel_regularizer=self.reg()))
			model.add(BatchNormalization(momentum=0.8))
		model.add(Dense(units = self.img_rows*self.img_cols*self.n_channels,activation = activation_fn, kernel_regularizer=self.reg()))
		model.add(Reshape(self.img_shape))		
		print("="*20+"Generator Layer"+"="*20)
		print(model.summary())
		noise = Input(shape=self.noise_shape)
		img = model(noise)
		
		return Model(noise,img)
		
	def build_discriminator(self):
		model = Sequential()
		#model.add(InputLayer(input_shape=(self.img_rows,self.img_cols)))
		model.add(Flatten(input_shape=self.img_shape))
		for i in range(len(self.discriminator_layers)):
			activation_fn = "sigmoid" if (i == len(self.discriminator_layers)-1) else "relu"
			model.add(Dense(units = self.discriminator_layers[i], activation = activation_fn, kernel_regularizer=self.reg()))	
		model.add(Dense(units = 1, activation = activation_fn, kernel_regularizer=self.reg()))
		print("="*10+"Discriminator Layer"+"="*10)
		print(model.summary())
		img = Input(shape=self.img_shape)
		validity = model(img)
		return Model(img,validity)
		
		
	def build_gan(self,training_generator=None):
		#Create Adam optimizer
		adam_optimizer = Adam(lr=self.learning_rate)
		
		#build discriminator
		discriminator_model = self.build_discriminator()
		discriminator_model.compile(loss=self.loss_fn, optimizer=adam_optimizer,metrics=["accuracy"])
		
		#build generator
		generator_model = self.build_generator()
		
		#Generator generates input from noise
		z = Input(shape=self.noise_shape)
		img = generator_model(z)
		
		#Compile model
		discriminator_model.trainable = False
		valid = discriminator_model(img)
		
		'''Combined model  (stacked generator and discriminator) takes noise as input => generates images => determines validity
        '''
		combined_model = Model(z, valid)
		combined_model.compile(loss=self.loss_fn, optimizer=adam_optimizer)
		
		return discriminator_model,generator_model, combined_model
		
	def generate_and_save_images(self,generator):
		if not os.path.exists(self.result_dir):
			os.makedirs(self.result_dir)
		for batch in range(self.batch_num):
			noise = np.random.normal(0, 1, (self.plot_rows*self.plot_cols, self.latent_dim))
			gen_imgs = generator.predict(noise)
		
			print(gen_imgs.shape)

			# Rescale images 0 - 1
			gen_imgs = (1/2.5) * gen_imgs + 0.5
		
			fig,axs = plt.subplots(self.plot_rows,self.plot_cols)
			idx = 0
			for i in range(self.plot_rows):
				for j in range(self.plot_cols):
					axs[i,j].imshow(gen_imgs[idx,])
					axs[i,j].axis("off")
					idx += 1
			fig.savefig("%s/img_%d.png" % (self.result_dir,batch))
			plt.close()


		
	def train_gan(self, discriminator, generator, combined_model):
		for i in range(self.epochs):
			#Adversarial ground truth
			valid = np.ones((self.batch_size,1))
			fake = np.zeros((self.batch_size,1))
			self.training_batches = int(math.ceil(self.x_train.shape[0]*1.0/self.batch_size))
			print("Epoch %d out of %d" % ((i+1),self.epochs))
			for i in range(self.training_batches):
				'''
				if i > 0:
					break
				'''
				last_idx = self.x_train.shape[0] if i == (self.training_batches-1) else (i+1)*self.batch_size
				imgs = self.x_train[i*self.batch_size:last_idx,]
				#Generate a batch of new images with generator
				noise = np.random.normal(0,1,(self.batch_size,self.latent_dim))
				gen_imgs = generator.predict(noise)
				#Train the discriminator
				disc_loss_real = discriminator.train_on_batch(imgs,valid)
				disc_loss_fake = discriminator.train_on_batch(gen_imgs,fake)
				disc_loss = 0.5*np.add(disc_loss_real,disc_loss_fake)
			
				#Train the generator
				noise = np.random.normal(0,1,(self.batch_size,self.latent_dim))
				gen_loss = combined_model.train_on_batch(noise,valid)
			
				 # Plot the progress
				print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (i, disc_loss[0], 100*disc_loss[1], gen_loss))
			
		return generator
			
			
			

if __name__=="__main__":
	synthesis_obj = ImageSynthesis()
	#X = synthesis_obj.load_metadata()
	#print(X[0:5])
