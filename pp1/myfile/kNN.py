import numpy as np 
import sys
import math
from operator import itemgetter
#import matplotlib.pyplot as plt

# Reading Data Files in arff format, return a 2D matrix which store the data
def read_file(filename):
	file_data = []
	with open(filename, 'r') as f:
		for line in f:
			if line[0].isdigit():
				features = []
				for data in line.split(','):
					try:
						features.append(float(data))
					except:
						features.append(data)
				file_data.append(features)
	return file_data

# calculate Euclidean distance
# don't use the class label
def calculate_dis(data1, data2):
	n = len(data1) - 1
	sum = 0.0
	i = 0
	while i < n:
		sum = sum + math.pow(data1[i] - data2[i], 2)
		i = i + 1
	return math.sqrt(sum)

# Implementing kNN algorithm
def kNN(train_set, test_set, k):
	accurate = 0
	for te in test_set:
		neighbors = []
		for tr in train_set:
			if len(neighbors) < k:
				distance = calculate_dis(te, tr)
				neighbors.append([tr, distance])
			else:
				temp = sorted(neighbors, key = lambda x: x[1])
				distance = calculate_dis(te, tr)
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

		sorted_label_count = sorted(label_count.items(), key = lambda x: x[1])

		if te[-1] == sorted_label_count[-1][0]:
			accurate = accurate + 1

	return float(accurate) / len(test_set) * 100

def main():
	#accuracy_list = []
 	#train_set = read_file(sys.argv[1])
 	#test_set = read_file(sys.argv[2])
 	train_files = ['spambase_train.arff', 'irrelevant_train.arff', 'mfeat-fourier_train.arff', 'ionosphere_train.arff']
 	test_files = ['spambase_test.arff', 'irrelevant_test.arff', 'mfeat-fourier_test.arff', 'ionosphere_test.arff']
 	i = 0
 	while i < 4:
 		train_set = read_file(train_files[i])
 		test_set = read_file(test_files[i])
 		accuracy_list = []
		for k in range(1,26):
			accuracy = kNN(train_set, test_set, k)
			accuracy_list.append(float(accuracy) * 0.01)
		print "accuracy is: "
		print accuracy_list
		i += 1

main()







