import numpy as np
import matplotlib.pyplot as plt


train_data1 = np.load('data1.npy')
train_lab1 = np.load('lab1.npy')

i = 3
plt.imshow(train_data1[i])
plt.savefig('img.png')
print(train_lab1[i])