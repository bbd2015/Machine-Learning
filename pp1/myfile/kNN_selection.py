import numpy as np 
import math
import sys
from operator import itemgetter

# read file
def read_file(filename):
	file_data = []
	with open(filename, 'r') as f:
		for line in f:
			if line[0].isdigit():
				feature = []
				for data in line.split(','):
					try:
						feature.append(float(data))
					except:
						feature.append(data)
				file_data.append(feature)
	return file_data

# calculate weighted distance
def calculate_weighted_dis(data1, data2, w):
	sum = 0.0
	n = len(data1) - 1
	i = 0
	while i < n:
		sum = sum + w[i] * math.pow(data1[i] - data2[i], 2)
		i += 1
	return math.sqrt(sum)

def kNN(train_set, test_set, k, w):
	accurate = 0
	for te in test_set:
		neighbors = []
		for tr in train_set:
			if len(neighbors) < k:
				distance = calculate_weighted_dis(te, tr, w)
				neighbors.append([tr, distance])
			else:
				temp = sorted(neighbors, key = lambda x: x[1])
				distance = calculate_weighted_dis(te, tr, w)
				if distance < temp[-1][1]:
					del temp[-1]
					temp.append([tr, distance])
				neighbors = temp

 		label_count = {}
		for neighbor in neighbors:
			label = neighbor[0][-1]
			t = True
			for (l, i) in label_count.items():
				if label == l:
					label_count[l] += 1
					t = False
					break
			if t == True:
				label_count[label] = 1

		sorte_label_count = sorted(label_count.items(), key = itemgetter(1))
		if te[-1] == sorte_label_count[-1][0]:
			accurate = accurate + 1

	return float(accurate) / len(test_set) * 100


def calculate_entropy(train_set):
	number_of_examples = len(train_set)
	total_freq = {}
	interval = int(number_of_examples / 5)
	entropy = []

	i = 0
	while i < 4 * interval:
		val_freq = {}
		data_entropy = 0.0
		for tr in train_set[i: i + interval]:
			if tr[-1] in val_freq:
				val_freq[tr[-1]] += 1.0
			else:
				val_freq[tr[-1]] = 1.0
		for freq in val_freq.values():
			data_entropy += (-freq/interval) * math.log(freq / interval, 2)
		entropy.append(data_entropy)
		i += interval

	while i < number_of_examples:
		val_freq = {}
		data_entropy = 0.0
		for tr in train_set[i:]:
			if tr[-1] in val_freq:
				val_freq[tr[-1]] += 1.0
			else:
				val_freq[tr[-1]] = 1.0

		for freq in val_freq.values():
			data_entropy += (-freq / len(train_set[i:])) * math.log(freq / len(train_set[i:]), 2)
		entropy.append(data_entropy)
		i = number_of_examples
	
	feature_entropy = 0.0
	for tr in train_set:
		if tr[-1] in total_freq:
			total_freq[tr[-1]] += 1.0
		else:
			total_freq[tr[-1]] = 1.0
	for fre in total_freq.values():
		feature_entropy += (-fre / number_of_examples) * math.log(fre / number_of_examples, 2)
	# calculate feature entropy
	return feature_entropy - sum(entropy) / 5.0



def main():
	train_files = ['spambase_train.arff', 'irrelevant_train.arff', 'mfeat-fourier_train.arff', 'ionosphere_train.arff']
 	test_files = ['spambase_test.arff', 'irrelevant_test.arff', 'mfeat-fourier_test.arff', 'ionosphere_test.arff']
 	index = 0
 	while index < 4:
 		#train_set = read_file(sys.argv[index])
 		#test_set = read_file(sys.argv[index])
		#k = int(sys.argv[3])
		train_set = read_file(train_files[index])
		test_set = read_file(test_files[index])
		k = 5
		number_of_features = len(train_set[0]) - 1
		print number_of_features
		print "\n"
		w = []
	
		for i in range(0,number_of_features):
			w.append([i,0])

		for j in range(0, number_of_features):
			sorted_train = sorted(train_set, key = itemgetter(j), reverse = True)
			information_gain = calculate_entropy(sorted_train)
			w[j][1] = information_gain

		sorted_w = sorted(w, key = itemgetter(1), reverse = True)
		accuracy_list = [] 
		i = 1
		while i <= number_of_features:
			weight = []
			for m in range(0,number_of_features):
				weight.append(0.0)
			for n in sorted_w[0:i]:
				weight[n[0]] = n[1]
			accuracy = kNN(train_set, test_set, k, weight)
			accuracy_list.append(float(accuracy) * 0.01)
			i += 1
		print accuracy_list
		print "\n"
		index += 1
main()
