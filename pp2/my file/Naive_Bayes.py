

import os
import math
import sys
import numpy as np
from collections import OrderedDict
#import matplotlib.pyplot as plt

# read index file
def read_index(index_file):
	file_list = OrderedDict()
	with open(index_file, 'r') as f:
		for line in f:
			line = line.split("|")
			file_list[line[0]] = line[1]
	return file_list


# read index split file
def read_index_split(split_file):
	file_yes = []
	file_no = []
	for key in split_file.keys():
		if split_file[key] == "yes":
			file_yes.append(key + ".clean")
		else:
			file_no.append(key + ".clean")

	return file_yes, file_no

# read train file and parse them into vocab
def read_file(train_file_yes, train_file_no):
	vocab_type1 = {}
	vocab_type2 = {}
	count_words_yes = 0
	count_words_no = 0
	for filename in train_file_yes:
		word_list = open(filename, 'r').read().split()
		count_list = [] # to calculate number of different files for vocab_type2
		for word in word_list:
			count_words_yes += 1
			if word not in vocab_type1:
				vocab_type1[word] = [1, 0]
			else:
				vocab_type1[word][0] += 1
				
			if word not in vocab_type2:
				vocab_type2[word] = [1, 0]
				count_list.append(word)
			else:
				if word not in count_list:
					vocab_type2[word][0] += 1
					count_list.append(word)
	#print vocab_type2

				
	for filename in train_file_no:
		word_list = open(filename, 'r').read().split()
		count_list = []
		for word in word_list:
			count_words_no += 1
			if word not in vocab_type1:
				vocab_type1[word] = [0, 1]
			else:
				vocab_type1[word][1] += 1
			if word not in vocab_type2:
				vocab_type2[word] = [0, 1]
				count_list.append(word)
			else:
				if word not in count_list:
					vocab_type2[word][1] += 1
					count_list.append(word)

	#print vocab_type2
	return vocab_type1, vocab_type2, count_words_yes, count_words_no
					
# calculate probability for every word with smoothing
def calculate_prob(vocab, m, number_yes, number_no, v):
	for key in vocab.keys():
		vocab[key][0] = (vocab[key][0] + m) / float(number_yes + m * v)
		vocab[key][1] = (vocab[key][1] + m) / float(number_no + m * v)
	return vocab

# predict test file class using training with type1 variant
def pred_label_type1(test_file, vocab, p):
	pred_yes = []
	pred_no = []
	for filename in test_file:
		pos_score = math.log(p['yes'])
		neg_score = math.log(p['no'])
		word_list = open(filename, 'r').read().split()
		for word in word_list:
			if word in vocab.keys():
				pos_score += np.log(vocab[word][0])
			if word in vocab.keys():
				neg_score += np.log(vocab[word][1])
		if pos_score > neg_score:
			pred_yes.append(filename)
		else:
			pred_no.append(filename)

	return pred_yes, pred_no

#predict test file class with type2 variant
def pred_label_type2(test_file, vocab, p):
	pred_yes = []
	pred_no = []
	for filename in test_file:
		pos_score = math.log(p['yes'])
		neg_score = math.log(p['no'])
		word_list = open(filename, 'r').read().split()
		for key in vocab.keys():
			if key in word_list:
				pos_score += np.log(vocab[key][0])
			else:
				pos_score += np.log(1.0 - vocab[key][0])
			if key in word_list:
				neg_score += np.log(vocab[key][1])
			else:
				neg_score += np.log(1.0 - vocab[key][1])
		if pos_score > neg_score:
			pred_yes.append(filename)
		else:
			pred_no.append(filename)
	return pred_yes, pred_no


def main():
	# first argument is index file
	# read train index file
	train_file = read_index(sys.argv[1])
	
	N = len(train_file)
	accuracy_type1_list = []
	accuracy_type2_list = []
	splits = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
	###############
	#m_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10]
	###############

	for split in splits:
	#for m in m_list:

		train_size = int(round(N * split))
		split_file = OrderedDict()
		for k in train_file.keys()[:train_size]:
			split_file[k] = train_file[k]
		train_file_yes, train_file_no = read_index_split(split_file)
		
		#train_file_yes, train_file_no = read_index_split(train_file)
		# calculate probability of 'yes' and 'no'
		train_len_yes = len(train_file_yes)
		train_len_no = len(train_file_no)
		train_len = train_len_yes + train_len_no
		p = {}
		p['yes'] = float(train_len_yes) / float(train_len)
		p['no'] = 1 - p['yes']

		#construct vocabulary
		# vocab_type1 = {"word": [# of words in yes class, # of words in no class}
		# vocab_type2 = {"word": [#of documents words in yes class, # of documents words in no class]}
	
		vocab_type1, vocab_type2, count_words_yes, count_words_no = \
			read_file(train_file_yes, train_file_no)
	
		# calculate probability for every word
		m = 1 # for test
		v_type1 = len(vocab_type1.keys())
		v_type2 = 2
		vocab_type1_prob = calculate_prob(vocab_type1, m, count_words_yes, count_words_no, v_type1)
		vocab_type2_prob = calculate_prob(vocab_type2, m, train_len_yes, train_len_no, v_type2)

		# read test index file
		test_file = read_index(sys.argv[2])
		test_file_yes, test_file_no = read_index_split(test_file)
		test_file = test_file_yes + test_file_no
		test_len = len(test_file)
		print "test file length: %d"%test_len
		# predict 'yes' and 'no' files
		test_pred_yes_type1, test_pred_no_type1 = \
			pred_label_type1(test_file, vocab_type1_prob, p)
		test_pred_yes_type2, test_pred_no_type2 = \
			pred_label_type2(test_file, vocab_type2_prob, p)


		#calculate prediction accuracy	
		accurate_type1 = 0
		accurate_type2 = 0
		for filename in test_file_yes:
			if filename in test_pred_yes_type1:
				accurate_type1 += 1
			if filename in test_pred_yes_type2:
				accurate_type2 += 1
		for filename in test_file_no:
			if filename in test_pred_no_type1:
				accurate_type1 += 1
			if filename in test_pred_no_type2:
				accurate_type2 += 1
	
		accuracy_type1 = float(accurate_type1) / float(test_len)
		accuracy_type2 = float(accurate_type2) / float(test_len)
		accuracy_type1_list.append(accuracy_type1)
		accuracy_type2_list.append(accuracy_type2)

	print "m = %d" %m
	print accuracy_type1_list
	print accuracy_type2_list
	
main()

