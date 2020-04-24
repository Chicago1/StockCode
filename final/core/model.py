import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import regularizers

class Model():
	"""A class for an building and inferencing an lstm model"""

	def __init__(self):
		self.model = Sequential()

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		self.model = load_model(filepath)

	def build_model(self, configs):

		for layer in configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None
			recurrent_reg_l1 = layer['recur_l1'] if 'recur_l1' in layer else 0.0
			recurrent_reg_l2 = layer['recur_l2'] if 'recur_l2' in layer else 0.0
			kernel_reg_l1 = layer['kernel_l1'] if 'kernel_l1' in layer else 0.0
			kernel_reg_l2 = layer['kernel_l2'] if 'kernel_l2' in layer else 0.0
			r_d = layer['recurrent_dropout'] if 'recurrent_dropout' in layer else 0.0

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation, 
                                    kernel_regularizer=regularizers.l1_l2(l1=kernel_reg_l1 , l2=kernel_reg_l2)))
			if layer['type'] == 'lstm':
				self.model.add(LSTM(neurons, input_shape=(input_timesteps, input_dim), return_sequences=return_seq, 
                                    recurrent_regularizer=regularizers.l1_l2(l1=recurrent_reg_l1 , l2=recurrent_reg_l2),
                                    kernel_regularizer=regularizers.l1_l2(l1=kernel_reg_l1 , l2=kernel_reg_l2),
                                    recurrent_dropout=r_d))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(dropout_rate))

		self.model.compile(loss=configs['model']['loss'], optimizer=configs['model']['optimizer'])

		print('[Model] Model Compiled')

	def train(self, x, y, v, epochs, batch_size, save_dir):
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
			ModelCheckpoint(filepath=save_fname, monitor='loss', save_best_only=True)
		]
		h = self.model.fit(
			x,
			y,
            validation_data = v,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)
        
		return(h)