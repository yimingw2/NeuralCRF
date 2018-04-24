import tensorflow as tf
import numpy as np
import argparse


class ProcessData():

	def __init__(self, train_data_path, dev_data_path):
		self.train_data_path = train_data_path
		self.dev_data_path = dev_data_path
		self.label = ["I-LOC", "B-LOC", "I-PER", "B-PER", "I-ORG", "B-ORG", "I-MISC", "B-MISC", "O"]

		self._build_dataset()


	def _build_dataset(self):
		"""
		build training and development dataset
		:param:
		:return words:
		"""
		train_x_raw, train_y_raw, train_vocab = self._load_data(self.train_data_path)
		dev_x_raw, dev_y_raw, dev_vocab = self._load_data(self.dev_data_path)
		self.vocab = train_vocab.union(dev_vocab)
		self.vocab.add("<unk>")
		self.vocab2idx, self.label2idx = self._build_vocab2idx()
		self.train_x, self.train_y, self.train_max_length = self._build_data_index(train_x_raw, train_y_raw)
		self.dev_x, self.dev_y, self.dev_max_length = self._build_data_index(dev_x_raw, dev_y_raw)


	def _load_data(self, data_path):
		"""
		load training data
		:param train_data_path: training data file
		:return words: a list of training instances, each contains a sentences with words
		:return labels: a list of corresponding labels
		:return vocab: a vocabulary set
		"""
		vocab = set()
		words = list()
		labels = list()

		with open(data_path) as train_data:
			f = iter(train_data)
			word_x = list()
			label_y = list()
			for line in f:
				if line == "\n":
					words.append(word_x)
					labels.append(label_y)
					word_x = list()
					label_y = list()
				else:
					parts = line.split()
					vocab.add(parts[0])
					word_x.append(parts[0])
					label_y.append(parts[3])
		return words, labels, vocab


	def _build_vocab2idx(self):
		"""
		build vocabulary to index from a vocabulary list
		:param vocab: vocabulary list
		:return vocab2idx: a dictionary from single word to index
		"""
		vocab2idx = dict()
		self.vocab = sorted(list(self.vocab))
		for i, item in enumerate(self.vocab):
			vocab2idx[item] = i + 1

		label2idx = dict()
		self.label = sorted(self.label)
		for i, item in enumerate(self.label):
			label2idx[item] = i

		return vocab2idx, label2idx


	def _build_data_index(self, data_x_raw, data_y_raw):
		"""
		build train and dev data from words to index
		:param data_x_raw: inputs word string
		:param data_y_raw: target label string
		:return data_x: inputs word index
		:return data_y: target label index
		"""
		max_length = 0

		data_x = list()
		for x in data_x_raw:
			max_length = max(max_length, len(x))
			data_x_i = list()
			for x_i in x:
				if x_i in self.vocab2idx:
					data_x_i.append(self.vocab2idx[x_i])
				else:
					data_x_i.append(self.vocab2idx["<unk>"])
			data_x.append(data_x_i)

		data_y = list()
		for y in data_y_raw:
			data_y_i = list()
			for y_i in y:
				data_y_i.append(self.label2idx[y_i])
			data_y.append(data_y_i)

		return data_x, data_y, max_length



class NeuralCRF():
	"""
	Defines a Neural CRF model
	:param : 
	:return :
	"""

	def __init__(self, vocab_list, label_list, embed_dim, bilstm_hidden_dim, batch_size, learning_rate, use_crf):

		# vocabulary and label list
		self.vocab_list = vocab_list
		self.label_list = label_list
		# scalar parameters
		self.embed_dim = embed_dim
		self.bilstm_hidden_dim = bilstm_hidden_dim
		self.batch_size = batch_size
		self.learning_rate = learning_rate
		# input and label
		self.input_raw = tf.placeholder(tf.int32, shape=[None, None], name='input')
		self.label_raw = tf.placeholder(tf.int32, shape=[None, None], name='label')
		# set up the initializer
		self.initializer = tf.contrib.layers.xavier_initializer()
		# build the network
		self._build_model(use_crf)
		# get loss and train_op
		if use_crf:
			self._crf()
			self.loss = self._crf_loss()
			self.accuracy = self._crf_accuracy()
		else: # only with bi-lstm
			self.loss = self._bilstm_loss()
			self.accuracy = self._bilstm_accuracy()
		self.train_op = self._train()


	def _build_model(self, use_crf):
		"""
		build model
		:param:
		:return:
		"""
		with tf.variable_scope("neural-crf"):
			# build embedding and input and one-hot target label
			if use_crf:
				emb = len(self.vocab_list) + 3
			else:
				emb = len(self.vocab_list) + 1 
			embedding_nobias = tf.get_variable(name="embedding", shape=[emb, self.embed_dim], dtype=tf.float32, initializer=self.initializer) # +1 is for 0 index
			zero_bias = tf.zeros([1, self.embed_dim], dtype=tf.float32)
			self.embedding = tf.concat([zero_bias, embedding_nobias], axis=0)
			self.input = tf.nn.embedding_lookup(params=self.embedding, ids=self.input_raw)
			self.label = tf.one_hot(self.label_raw, depth=len(self.label_list), dtype=tf.int32)
			# build input sequence length
			count_nonzero = tf.count_nonzero(self.input, axis=2)
			self.input_length = tf.cast(tf.count_nonzero(count_nonzero, axis=1), dtype=tf.int32) # [batch_size,]
			self._bilstm()


	def _bilstm(self):
		"""
		define bi-lstm model
		:param:
		:return lstm_outputs: [batch_size, max_time, label_size]
		"""
		with tf.variable_scope("bi-lstm"):
			bilstm_cell = dict()
			for direction  in ["f", "b"]:
				with tf.variable_scope(direction):
					bilstm_cell[direction] = tf.contrib.rnn.LSTMCell(self.bilstm_hidden_dim, initializer=self.initializer)
			outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=bilstm_cell["f"], cell_bw=bilstm_cell["b"], \
														 inputs=self.input, sequence_length=self.input_length, dtype=tf.float32)
			self.outputs = tf.concat(outputs, 2)


	def _crf(self):
		"""
		define bi-lstm-crf model
		:param:
		:return crf_outputs: [batch_size, max_time, label_size]
		"""
		with tf.variable_scope("crf"):
			self.transition = tf.get_variable(name="transition", shape=[len(self.label_list)+2, len(self.label_list)+2], \
											  dtype=tf.float32, initializer=self.initializer) # +2 for <s> and <\s>
			self.lstm_outputs_with_start_end = tf.contrib.layers.fully_connected(self.outputs, len(self.label_list)+2, activation_fn=None)
			self.crf_log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(inputs=self.lstm_outputs_with_start_end, \
																	  tag_indices=self.label_raw, \
																	  sequence_lengths=self.input_length, \
																	  transition_params=self.transition)


	def _bilstm_loss(self):
		"""
		calculate bi-lstm loss
		:param: None
		:return loss:
		"""
		lstm_outputs = tf.contrib.layers.fully_connected(self.outputs, len(self.label_list), activation_fn=None)
		self.output_softmax = tf.nn.softmax(lstm_outputs, axis=2) # softmax layer
		label_temp = tf.cast(self.label, dtype=tf.float32)
		cross_entropy_total = -tf.reduce_sum(label_temp * tf.log(self.output_softmax), axis=2)
		label_length_temp = tf.cast(self.input_length, dtype=tf.float32)
		cross_entropy = tf.reduce_sum(cross_entropy_total, axis=1) / label_length_temp
		loss = tf.reduce_mean(cross_entropy)
		return loss


	def _bilstm_accuracy(self):
		"""
		calculate bi-lstm accuracy
		:param: None
		:return accuracy:
		"""
		incorrect_num = tf.count_nonzero(self.label_raw - tf.argmax(self.output_softmax, axis=2, output_type=tf.int32), dtype=tf.int32)
		incorrect_num = tf.cast(incorrect_num, dtype=tf.float32)
		total_num = tf.cast(tf.reduce_sum(self.input_length), dtype=tf.float32)
		correct_num = total_num - incorrect_num
		accuracy = correct_num / total_num
		return accuracy


	def _crf_loss(self):
		"""
		calculate crf loss
		:param: None
		:return loss:
		"""
		loss = -tf.reduce_mean(self.crf_log_likelihood)
		return loss


	def _crf_accuracy(self):
		"""
		calculate crf accuracy
		:param: None
		:return accuracy:
		"""
		# #### TODO: this should only be used at test time, and the batch size should be 1 ####
		# self.score_nobatch = tf.squeeze(input=self.lstm_outputs_with_start_end, axis=0)
		# decode_seq, _ = tf.contrib.crf.viterbi_decode(score=score_nobatch, transition_params=self.transition)
		# label_nobatch = tf.squeeze(input=self.label_raw, axis=0)
		# incorrect_num = tf.count_nonzero(self.label_nobatch - decode_seq, dtype=tf.int32)
		# total_num = tf.cast(tf.reduce_sum(self.input_length), dtype=tf.float32)
		# correct_num = total_num - incorrect_num
		# # accuracy = correct_num / total_num
		# return correct_num, total_num

		decode_seq, _ = tf.contrib.crf.crf_decode(potentials=self.lstm_outputs_with_start_end, \
												  transition_params=self.transition, \
												  sequence_length=self.input_length)
		incorrect_num = tf.count_nonzero(self.label_raw - decode_seq, dtype=tf.int32)
		incorrect_num = tf.cast(incorrect_num, dtype=tf.float32)
		total_num = tf.cast(tf.reduce_sum(self.input_length), dtype=tf.float32)
		correct_num = total_num - incorrect_num
		accuracy = correct_num / total_num
		return accuracy


	def _train(self):
		"""
		define optimizer
		:param: None
		:return train_op: 
		"""
		train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
		return train_op



class TrainModel():

	def __init__(self, dataset, params, sess):

		self.sess = sess
		self.dataset = dataset
		self.params = params
		self.model = NeuralCRF(dataset.vocab, dataset.label, params["embed_dim"], params["bilstm_hidden_dim"], params["batch_size"], params["learning_rate"], params["use_crf"])
		self.init = tf.global_variables_initializer()

		self.train_x, self.train_y = self._pad_data(dataset.train_x, dataset.train_y, dataset.train_max_length, params["use_crf"])
		self.dev_x, self.dev_y = self._pad_data(dataset.dev_x, dataset.dev_y, dataset.dev_max_length, params["use_crf"])


	def _pad_data(self, data_x, data_y, max_length, use_crf):

		data_x_new = list()
		data_y_new = list()
		if use_crf:
			max_length += 2
			sent_start_idx = len(self.dataset.vocab) + 1
			sent_end_idx = len(self.dataset.vocab) + 2
			label_start_idx = len(self.dataset.label)
			label_end_idx = len(self.dataset.label) + 1

		for i, sent in enumerate(data_x):
			if len(sent) == max_length:
				data_x_new.append(np.array(sent))
				data_y_new.append(np.array(data_y[i]))
			else:
				pad_num = max_length - len(sent)
				if use_crf:
					pad_num -= 2
					data_x_new.append(np.concatenate([np.array([sent_start_idx]), np.array(sent), np.array([sent_end_idx]), np.zeros([pad_num,])]))
					data_y_new.append(np.concatenate([np.array([label_start_idx]), np.array(data_y[i]), np.array([label_end_idx]), np.zeros([pad_num,])]))
				else:
					data_x_new.append(np.concatenate([np.array(sent), np.zeros([pad_num,])]))
					data_y_new.append(np.concatenate([np.array(data_y[i]), np.zeros([pad_num,])]))
		data_x_new = np.reshape(np.concatenate(np.array(data_x_new)), [len(data_x_new), -1])
		data_y_new = np.reshape(np.concatenate(np.array(data_y_new)), [len(data_y_new), -1])

		return data_x_new, data_y_new


	def _get_train_data(self):

		a = np.arange(self.train_x.shape[0])
		np.random.shuffle(a)
		train_x_new = self.train_x[a]
		train_y_new = self.train_y[a]

		return train_x_new, train_y_new


	def train(self):

		epoch = self.params["epoch"]
		self.sess.run(self.init)
		sz = self.params["batch_size"]
		l = self.train_x.shape[0]
		batch = int(l / sz)
		for i in range(epoch):
			train_sent, train_label = self._get_train_data()
			for j in range(batch):
				train_sent_feed = train_sent[j*sz:min((j+1)*sz, l)]
				train_label_feed = train_label[j*sz:min((j+1)*sz, l)]
				_, loss, acc = self.sess.run(fetches=[self.model.train_op, self.model.loss, self.model.accuracy], \
											 feed_dict={self.model.input_raw: train_sent_feed, self.model.label_raw: train_label_feed})
				if j % 50 == 0:	
					print("Training epoch {}, batch: {}, loss: {}, accuracy: {}".format(i, j, loss, acc))
			loss, acc = self.sess.run(fetches=[self.model.loss, self.model.accuracy], \
					 	  			  feed_dict={self.model.input_raw: self.dev_x, self.model.label_raw: self.dev_y})
			print("testing, loss: {}, accuracy: {}".format(loss, acc))

	

def parse_arguments():

	parser = argparse.ArgumentParser(description="Neural CRF")
	parser.add_argument('--train-data', dest='train_data_path', default='train.data', type=str)
	parser.add_argument('--test-data', dest='test_data_path', default='dev.data', type=str)

	return parser.parse_args()


def training_params():

	params = dict()
	params["use_crf"] = True
	params["embed_dim"] = 50
	params["bilstm_hidden_dim"] = 300
	params["batch_size"] = 32
	params["learning_rate"] = 1e-2
	params["epoch"] = 4

	return params


def main():
	args = parse_arguments()
	train_data_path = args.train_data_path
	test_data_path = args.test_data_path

	dataset = ProcessData(train_data_path, test_data_path)
	params = training_params()
	sess = tf.Session()

	train_obj = TrainModel(dataset, params, sess)
	train_obj.train()


if __name__ == "__main__":
	main()
