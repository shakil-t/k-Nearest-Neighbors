# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 12:46:52 2019

@author: shakil
"""

from math import sqrt
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import random

def load_data(program_file_name, line):
    file=open(program_file_name, "r")
    read_to_scan=file.read()
    instances=read_to_scan.split("\n")
    data_set=[]
    for i in range(line, len(instances)-1):
        temp=instances[i].split(",")
        data_set.append(temp)
    random.shuffle(data_set)
    data_set=np.asarray(data_set, dtype=float)
    return data_set

def normalization(data_set):
    #we should normalize the numeric features and our standard would be x=(x-min)/(max-min)
    for value in numeric_features:
        column=data_set[:, value]
        maximum=max(column)
        minimum=min(column)
        for i in range(0, len(data_set)):
            data_set[i][value]=(data_set[i][value]-minimum)/(maximum-minimum)
    return data_set

#change the index due to your need 
def sort(l):
    l.sort(key=lambda x: x[19])
    return l

def find_neighbors(distance):
    classification=list(distance[:, len(distance[0])-1])
    lable=[]
    for value in classes:
        lable.append(classification.count(value))
    return lable.index(max(lable))

def create_distance_matrix(test_data, train_data, k):
    error=0
    for i in range(0, len(test_data)):
        distance=[]
        temp2=[]
        for j in range(0, len(train_data)):
            temp1=[]
            #or len(train_data[0]) cause they're equal
            for l in range(0, len(test_data[0])-1):
                if l in numeric_features:
                    temp1.append((test_data[i][l]-train_data[j][l])**2)
                else:
                    if test_data[i][l]!=train_data[j][l]:
                        temp1.append(1)
                    else:
                        temp1.append(0)
            temp1.append(sqrt(sum(temp1)))
            temp1.append(train_data[j][len(train_data[0])-1])
            temp2.append(temp1.copy())
        temp2=sort(temp2)
        distance=temp2[0: k]
        distance=np.asarray(distance)
        expected_class=find_neighbors(distance)+1
        if test_data[i][len(test_data[0])-1]!=expected_class:
            error+=1
    return error/len(test_data)

def knn(program_file_name, line, k, distance_metric="Euclidus"):
    #Let's run our algorithm
    data_set=load_data(program_file_name, line)
    data_set=normalization(data_set)
    
    #first fold from 0 to 206
    train_data1=data_set[207: , :]
    test_data1=data_set[0: 206, :]
    
    #second fold from 207 to 413
    index1=[i for i in range(0, 207)]
    index2=[i for i in range(413, 1034)]
    index=index1+index2
    train_data2=[data_set[i] for i in index]
    test_data2=data_set[207: 413, :]
    
    #third fold from 413 to 621
    index1=[i for i in range(0, 413)]
    index2=[i for i in range(621, 1034)]
    index=index1+index2
    train_data3=[data_set[i] for i in index]
    test_data3=data_set[413: 621, :]
    
    #forth fold from 621 to 829
    index1=[i for i in range(0, 621)]
    index2=[i for i in range(829, 1034)]
    index=index1+index2
    train_data4=[data_set[i] for i in index]
    test_data4=data_set[621: 829, :]
    
    #fifth fold from 829 to 1034
    train_data5=data_set[0: 829, :]
    test_data5=data_set[829: , :]
    
    for value in k:
        test_error=[]
        train_error=[]
        
        print("k=", value)
        
        print("First Fold:")
        train_error.append(create_distance_matrix(train_data1, train_data1, value))
        print("Train Error=", train_error[0])
        test_error.append(create_distance_matrix(test_data1, train_data1, value))
        print("Test Error=", test_error[0])
        print()
        
        
        print("Second Fold:")
        train_error.append(create_distance_matrix(train_data2, train_data2, value))
        print("Train Error=", train_error[1])
        test_error.append(create_distance_matrix(test_data2, train_data2, value))
        print("Test Error=", test_error[1])
        print()
        
        
        print("Third Fold:")
        train_error.append(create_distance_matrix(train_data3, train_data3, value))
        print("Train Error=", train_error[2])
        test_error.append(create_distance_matrix(test_data3, train_data3, value))
        print("Test Error=", test_error[2])
        print()
        
        
        print("Forth Fold:")
        train_error.append(create_distance_matrix(train_data4, train_data4, value))
        print("Train Error=", train_error[3])
        test_error.append(create_distance_matrix(test_data4, train_data4, value))
        print("Test Error=", test_error[3])
        print()
        
        
        print("Fifth Fold:")
        train_error.append(create_distance_matrix(train_data5, train_data5, value))
        print("Train Error=", train_error[4])
        test_error.append(create_distance_matrix(test_data5, train_data5, value))
        print("Test Error=", test_error[4])
        print()
        
        print("Mean Train Error=", sum(train_error)/len(train_error))
        print("Mean Test Error=", sum(test_error)/len(test_error))
        print()
        
        e_test.append(sum(test_error)/len(test_error))
        e_train.append(sum(train_error)/len(train_error))
        
    diagram()

#to plot the error
#figsize changes the size of the plot
#dpi helps you to increase or decrease the resolution
#change the font by fontsize
#check all the styles at https://matplotlib.org/3.1.0/gallery/style_sheets/style_sheets_reference.html
#find the hex and rgb code of your favorite color at https://www.rapidtables.com/web/color/
def diagram(style="dark_background"):
    mpl.style.use(style)
    fig, ax=plt.subplots(figsize=(9, 9), dpi=150)
    ax.set_title("Error", color='#7B68EE')
    ax.plot(k, e_test, '#6495ED', label='Mean Test Error')
    ax.plot(k, e_train,'#DC143C', label='Mean Train Error')
    ax.legend(fontsize=20)

#let's load the data. This time we won't use a dataframe as it's easier to handle it in a 2D nparray
program_file_name="ThyroidData.arff"
#According to our arff file data starts from line 24
line=24
classes=[1, 2, 3]
#the list below is the indices of numeric features to make our work easier
numeric_features=[0, 14, 15, 16, 17, 18]
k=[1, 10, 20, 50, 100, 200, 500, 800]
#e stands for error
e_test=[]
e_train=[]
knn(program_file_name, line, k, distance_metric="Euclidus")
