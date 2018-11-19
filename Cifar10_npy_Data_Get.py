# -*- coding: utf-8 -*-
"""
Created on Sun Oct  7 00:15:25 2018

@author: craig
"""
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time

t0 = time.clock()

def load_cfar10_batch(cifar10_dataset_folder_path, batch_id):
    with open(cifar10_dataset_folder_path + '\\data_batch_' + str(batch_id), mode='rb') as file:
        # 編碼方式採用'latin1'
        batch = pickle.load(file, encoding='latin1')#pickle拆解
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32))#reshape
    labels = batch['labels']
    #one_hot encoding
    for num in range(10000):
        index = labels[num]
        one_hot = np.zeros((1,10))
        one_hot[0][index] = 1
        labels[num] = one_hot
    return features, labels

def generate_data(batch_id):#獲得數據
    cifar10_dataset_folder_path = "C:\\Users\\craig\\OneDrive\\桌面\\prediction\\cifar-10-python\\cifar-10-batches-py"
    features, labels = load_cfar10_batch(cifar10_dataset_folder_path, batch_id)
    return features, labels

def print_pic(pic):#畫圖用(3D圖也可)
    plt.imshow(pic)
    
def transpose(matrix):#將矩陣的channel放到最後面
    T_matrix = np.zeros((len(matrix),32,32,3), dtype=int)
    for i in range(len(matrix)):
        T_matrix[i] = np.transpose(matrix[i], (1,2,0))
    return T_matrix

def get_test_batch():#圖裡上面的方法獲得test-set
    cifar10_dataset_folder_path = "C:\\Users\\craig\\OneDrive\\桌面\\prediction\\cifar-10-python\\cifar-10-batches-py"
    with open(cifar10_dataset_folder_path + '\\test_batch', mode='rb') as file:
        # 編碼方式採用'latin1'
        batch = pickle.load(file, encoding='latin1')
    features = batch['data'].reshape((len(batch['data']), 3, 32, 32))
    labels = batch['labels']
    #one_hot encoding
    for num in range(10000):
        index = labels[num]
        one_hot = np.zeros((1,10))
        one_hot[0][index] = 1
        labels[num] = one_hot
    return features, labels

def rgb2gray(image):#參考網路上的圖片轉灰階函數
    return np.dot(image[...,:3], [0.299, 0.587, 0.114])

def sobel_add(image, image_size):#image_size = [batch, height, width, channel]
    sobel_add_image = np.ones((1,32,32,5), np.int32)#一張一張圖疊起來的起始array
    for i in range(len(image)):#把每張圖拆開來處理
        gray_image = rgb2gray(image[i])#轉成灰階
        sobel_image = tf.image.sobel_edges(tf.convert_to_tensor(gray_image.reshape(1,32,32,1)))#邊緣檢測，回傳兩張圖(垂直、水平)
        sobel_res = tf.transpose(sobel_image,(4,0,1,2,3))#排列傳回來的圖，把batch放到第一個參數
        sobel_res = tf.reshape(tf.concat([image[i], tf.reshape(sobel_res, [32,32,2])], 2), [1,32,32,5])#把邊緣檢測的圖加到原本的圖上變成五通道
        sobel_add_image = tf.concat((sobel_add_image, tf.cast(sobel_res, dtype=tf.int32)), 0)#每張圖處理完整理成同一組batch
        if i % 1000 == 0 and i != 0:
            print("已經加載",i,"筆數據!","這組數據剩餘",5000-i,"筆資料須加載!")
    return sobel_add_image.eval()[1:]#除了第一組其他回傳

def save_to_npy(X_Train, X_Train_with_Sobel, Y_Train, X_Test, X_Test_with_Sobel, Y_Test, Label_Mean):#save所有獲得的資料存成npy檔
    np.save("Cifar10_X_Train.npy",X_Train)
    np.save("Cifar10_X_Train_with_Sobel.npy",X_Train_with_Sobel)
    np.save("Cifar10_Y_Train.npy",Y_Train)
    np.save("Cifar10_X_Test.npy",X_Test)
    np.save("Cifar10_X_Test_with_Sobel.npy",X_Test_with_Sobel)
    np.save("Cifar10_Y_Test.npy",Y_Test)
    np.save("Cifar10_Label_Mean.npy",Label_Mean)
    

#--------------------------------------------------------------------------------------------------
sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

"""
定義訓練集、測試集
"""
print("開始加載數據...")
X_Train, Y_Train = generate_data(1)
for num_of_batch in range(2,6):#從batch1~5獲得資料
    X_Train = np.concatenate((X_Train,generate_data(num_of_batch)[0]),0)
    Y_Train = np.concatenate((Y_Train,generate_data(num_of_batch)[1]),0)
X_Train = transpose(X_Train)
Y_Train = np.array(Y_Train,dtype=np.int32).reshape(-1,10)#簡單的reshape處理

X_Test_Temp, Y_Test = get_test_batch()
X_Test = transpose(X_Test_Temp)
Y_Test = np.array(Y_Test,dtype=np.int32).reshape(-1,10)

Label_Mean = np.array(["Airplane","Automobile","Bird","Cat","Deer","Dog","Frog","Horse","Ship","Truck"])#結果label的真實意義
X_Train_with_Sobel = np.ones((1,32,32,5), np.int32)
for i in range(10):
    print("開始加載第", i+1, "組數據!", "剩餘", 10-i, "組數據須加載!")
    X_Train_with_Sobel = np.concatenate((X_Train_with_Sobel, sobel_add(X_Train[i*5000:(i+1)*5000], np.shape(X_Train[i*5000:(i+1)*5000]))), 0)#多加2層sobel的X_Train
X_Train_with_Sobel = X_Train_with_Sobel[1:]
print("開始加載測試集數據")
X_Test_with_Sobel = sobel_add(X_Test, np.shape(X_Test))#多加2層sobel的X_Test
print("稍等一下所有資料馬上加載完成...")
save_to_npy(X_Train, X_Train_with_Sobel, Y_Train, X_Test, X_Test_with_Sobel, Y_Test, Label_Mean)
print("所有資料加載完成!")
print("總共費時:",time.clock()-t0,"秒")
