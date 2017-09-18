import os
import matplotlib.pyplot as plt 

accuracy = [0.916785, 0.645, 0.746627, 0.91453]
x = [1,2,3,4]
labels = ['spambase', 'irrelevant', 'mfeat-fourier', 'ionosphere']

plt.plot(x,accuracy,'bs',markersize=4)
plt.plot(x,accuracy,'r')
plt.xticks(x, labels)
plt.xlabel("data sets")
plt.ylabel("Accuracy for test sets")
plt.title("Evaluation for J48 algorithm")
plt.grid(True)
plt.show()