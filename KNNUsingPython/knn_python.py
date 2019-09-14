#!/usr/bin/env python
# coding: utf-8

# In[3]:


# knn
import math
from collections import defaultdict
from operator import itemgetter

import numpy as np 
import matplotlib.pyplot as plt

warna=['red','blue','purple']

#------------------------- KNN FUNCTION -------------------------------------
def get_most_common_item(array):
    count_dict = defaultdict(int)
    for key in array:
        count_dict[key] += 1
    key, count = max(count_dict.items(), key=itemgetter(1))
    return key


# https://en.wikipedia.org/wiki/Euclidean_distance
def euclidean_dist(A, B):
    return math.sqrt(sum([(A[i]-B[i])**2 for i, _ in enumerate(A)]) )

def knn(X_train, y_train, X_test, k=1):
    y_test = []
    for test_row in X_test:
        eucl_dist = [euclidean_dist(train_row, test_row) for train_row in X_train]
        sorted_eucl_dist = sorted(eucl_dist)
        closest_knn = [eucl_dist.index(sorted_eucl_dist[i]) for i in range(0, k)] if k > 1 else [eucl_dist.index(min(eucl_dist))]
        closest_labels_knn = [y_train[x] for x in closest_knn]
        y_test.append(get_most_common_item(closest_labels_knn))
    return y_test

def gambarGrafik():
    plt.title('Classification KNN')
    plt.xlabel('Nilai x1') #Sumbu -X
    plt.ylabel('Nilai x2')#Sumbu -Y 
    idx=0
    for xy in X_train:
        plt.scatter(xy[0],xy[1],color=warna[y_train[idx]],s=10)
        idx+=1
    plt.show()
#---------------------------- END KNN FUNCTION -----------------------------------


# --------------------------- Kondisi awal dan Problem ---------------------------
# Diketahui Data train/test data 
X_train = [
    [1, 1],
    [1, 2],
    [2, 4],
    [3, 5],
    [1, 0],
    [0, 0],
    [1, -2],
    [-1, 0],
    [-1, -2],
    [-2, -2]
]

# nilai kelas (y_train) berdasarkan data x_train
y_train = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

# Berdasarkan data training x_train dan y_train diatas, 
# tentukan nilai kelas dari data testing x_test dibawah
X_test = [
    [5, 5],
    [0, -1],
    [-5, -5]
]

# menggambarkan grafik data x_train dan data x_test 
# dimana data kelas x_train berwarna biru jika nilai kelas y_train =1 
# dan ungu jika bernilai 2
# Sedangkan data x_test yang belum ditentukan kelasnya akan berwarna merah
print ('\033[1m' + "GRAFIK SEBELUM KLASIFIKASI")
idx=0
for xy in X_test:
    plt.scatter(xy[0],xy[1],color=warna[0])
    idx+=1

gambarGrafik()
print ("Masuk kelas manakah data X_TEST yang disimbolkan dengan lingkaran warna merah?")
print ()

print ("========================================================")

# Problem: Tentukan kelas untuk masing-masing data x_test menggunakan KNN 
# dimana nilai k=2 ??????
# JAWAB:
y_test= knn(X_train, y_train, X_test, k=2) # Memanggil fungsi Knn

print ("JAWAB: \nHASIL KELAS Y_TEST DARI DATA X_TEST=",y_test)
print ()

# menggambarkan grafik hasil klasifikasi
print ('\033[1m' + "GRAFIK SETELAH KLASIFIKASI")
idx=0
for xy in X_test:
    plt.scatter(xy[0],xy[1],color=warna[y_test[idx]])
    idx+=1

gambarGrafik()

# ------------------------ END Kondisi awal dan Problem --------------------------


# In[ ]:




