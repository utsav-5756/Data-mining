# -*- coding: utf-8 -*-
"""
Created on Tue Mar 20 19:19:06 2018

@author: utsav
"""


import sympy as sp
import numpy as np
import numpy.linalg as nl
import matplotlib.pyplot as plt

x = np.linspace(0,1,200)

y = np.zeros_like(x,dtype = np.int32)

x[0:100] = np.sin(4*np.pi*x)[0:100]

x[100:200] = np.cos(4*np.pi*x)[100:200]

y = 4*np.linspace(0,1,200)+0.3*np.random.randn(200)

label= np.ones_like(x)

label[0:100]=0

X = sp.Symbol('X')                                  #declaring a variable for matrix
Y = sp.Symbol('Y')                                  #declaring a variable for matrix

                                                    
    
mean_class_0_1 = np.mean(x[0:100])                  #mean of elements of x for class=0    
mean_class_0_2 = np.mean(y[0:100])                  #mean of elements of y for class=0
mean_class_1_1 = np.mean(x[100:200])                #mean of elements of x for class=1    
mean_class_1_2 = np.mean(y[100:200])                #mean of elements of y for class=1
    
mean_class_0 = np.matrix([mean_class_0_1, mean_class_0_2])  #mean matrix for class=0
mean_class_1 = np.matrix([mean_class_1_1,mean_class_1_2])   #mean matrix for class=1

def lda():                                                  #function LDA
    
    """covariance"""
    inverse_covariance = nl.inv(np.cov(x,y))                #inverse of covaraince
    
    A = sp.Matrix([[X,Y]])                                  #Declaring a variable matrix
    A_t = A.transpose()                                     #Transpose of matrix
    a_1= inverse_covariance*A_t
    a = (-mean_class_0 + mean_class_1)*(a_1)                
    
    b_1 = A*inverse_covariance
    b = b_1*(-mean_class_0.T + mean_class_1.T)
    
    c = mean_class_0*inverse_covariance*mean_class_0.T - mean_class_1*inverse_covariance*mean_class_1.T  #calculation for the multiplication of matrix
    c_f = np.asscalar(c)
    
    Z = a[0] + b[0] + c_f
    Z = Z*-1                                    #decision boundary for LDA
    print("Decision boundary for LDA is\n",Z)       #printing the decision boundary
  
    for_X = Z.subs({X:-1.1})
    Y_value = sp.solve(for_X,Y)
        
    for_Y = Z.subs({Y:1.5})
    X_value = sp.solve(for_Y,X)
    
    p1 = [-1.1,X_value[0]]
    p2 = [Y_value[0],1.5]
    plt.plot(p1,p2)                             #plotting the decision boundary
    plt.scatter(x,y,c=label)
    plt.show()
    
def qda():                                      #declaring the QDA function
    
    """covariance"""
    inverse_covariance_class_0 = nl.inv(np.cov(x[0:100],y[0:100]))          #inverse of covariance for class=0
    inverse_covariance_class_1 = nl.inv(np.cov(x[100:200],y[100:200]))      #inverse of covariance for class=1
    
    value_0 = np.matrix(np.log(np.linalg.det(np.cov(x[0:100],y[0:100]))))
    value_1 = np.matrix(np.log(np.linalg.det(np.cov(x[100:200],y[100:200]))))
    
    A = sp.Matrix([[X,Y]])                                                  #matrix with variables
    
    a_1 = A*(inverse_covariance_class_0 - inverse_covariance_class_1)
    a = a_1*A.T

    
    b = (mean_class_0*inverse_covariance_class_0 - mean_class_1*inverse_covariance_class_1)*A.T
        
    c= A*(inverse_covariance_class_0*mean_class_0.T - inverse_covariance_class_1*mean_class_1.T)
        
    d = (mean_class_0*inverse_covariance_class_0*mean_class_0.T) - (mean_class_1*inverse_covariance_class_1*mean_class_1.T)
    
    e = value_0 - value_1
    
    Z = a[0] - b[0] - c[0] - d + e                              #calculating the decision boundary
    Z = sp.simplify(Z)
    print("Decision boundary for QDA is\n",Z)                   #printing the decision boundary
    
    x_ax = np.linspace(-3,3,20)                                 
    y_ax = np.linspace(-3,3,20)
    
    x_g,y_g = np.meshgrid(x_ax,y_ax)                            #creating a grid with some points 
    
    def f(p,q):                                                 #function to calculate the value of decision boundary for points
        sub = Z.subs({X:p,Y:q})
        sub1 = np.array(sub)
        sub2 = sub1.item()
        return sub2                                             #returning the calculated value
    
    
    fn = []
    for k in range(20): 
        final = []
        for l in range(20):
            var = f(x_g[k][l],y_g[k][l])
            final.append(var)
        fn.append(final)                                        #storing the final values in an array to plot the decision boundary for QDA
    plt.scatter(x,y,c=label)
    plt.contour(x_g, y_g, fn,[30])                              #plotting the decision boundary for QDA
    plt.show()
    

lda()                                                           #calling the LDA function
qda()                                                           #calling the QDA function

