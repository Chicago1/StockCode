import os
import math
import numpy as np
import datetime as dt
from numpy import newaxis
from tensorflow.keras.layers import Dense, Activation, Dropout, LSTM
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from kerastuner import HyperModel

from tensorflow import keras

class LSTMHyperModel(HyperModel):
	"""A class for an building and inferencing an lstm model"""

	def __init__(self, configs):
		self.model = None
		self.configs = configs

	def load_model(self, filepath):
		print('[Model] Loading model from file %s' % filepath)
		model = load_model(filepath)

	def build(self, hp):
		self.model = keras.Sequential()

		for layer in self.configs['model']['layers']:
			neurons = layer['neurons'] if 'neurons' in layer else None
			dropout_rate = layer['rate'] if 'rate' in layer else None
			activation = layer['activation'] if 'activation' in layer else None
			return_seq = layer['return_seq'] if 'return_seq' in layer else None
			input_timesteps = layer['input_timesteps'] if 'input_timesteps' in layer else None
			input_dim = layer['input_dim'] if 'input_dim' in layer else None

			if layer['type'] == 'dense':
				self.model.add(Dense(neurons, activation=activation)) #leave this one alone for now
			if layer['type'] == 'lstm':
				self.model.add(LSTM(units=hp.Int(
									'neurons',
									min_value=50,
									max_value=300,
									step=50,
									default=neurons),
								input_shape=(input_timesteps, input_dim),
								return_sequences=return_seq))
			if layer['type'] == 'dropout':
				self.model.add(Dropout(rate = hp.Float(
									'dropout_rate',
									min_value=0.0,
									max_value=0.5,
									step = 0.05,
									default=dropout_rate
				)))


		self.model.compile(loss=self.configs['model']['loss'], optimizer=hp.Choice(
			'optimizer',
			values=['sgd', 'adam'],
			default=self.configs['model']['optimizer']

		))

		print('[Model] Model Compiled')

		return self.model


	def train(self, x, y, epochs, batch_size, save_dir):
		print('[Model] Training Started')
		print('[Model] %s epochs, %s batch size' % (epochs, batch_size))
		
		save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
		callbacks = [
			EarlyStopping(monitor='val_loss', patience=2),
			ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True)
		]
		self.model.fit(
			x,
			y,
			epochs=epochs,
			batch_size=batch_size,
			callbacks=callbacks
		)
		self.model.save(save_fname)

		print('[Model] Training Completed. Model saved as %s' % save_fname)