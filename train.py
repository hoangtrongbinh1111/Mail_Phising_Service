from model import Model


async def train(data_dir,learning_rate, epochs, batch_size, val_size, model_type, labId):
	"""
	Train model

	Parameters:
	-----------
	data_dir: 	str,
				Training data directory.

	learning_rate: 	float,
					Learning rate for training model.
	epochs:	int,
			Number of training epochs.

	batch_size: int,
				Batch size of training data.
	val_size: 	float,
				Size of validation set over training dataset
	model_type: string,
				Type of rnn cells for building model

	labId:	string,
			ID of lab (use for backend)
	Returns:
	--------
	Trained models saved by .ckpt file
	"""
    
    #Call model from Model class for training

	model = Model(labId, model_type, train_data_dir = data_dir, val_size=val_size)
	train_output = model.train(learning_rate, epochs, batch_size)
	for res_per_epoch in train_output:
		yield res_per_epoch


if __name__ == '__main__':
	train("data", 0.001, 5, 10, 0.2, 'lstm', 'lab2')
