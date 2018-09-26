
"""
CS 641
Name: Utsav Shaileshkumar Chokshi
Date: 02/20/2018
"""

from sklearn.datasets import load_iris

data = load_iris()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(data['data'],data['target'], random_state = 0)




import numpy as np
import matplotlib.pyplot as plt
import math
import timeit
import time

"""Start of class"""

class KNN(object):
    def __init__(self):
        pass
    
    
    def train(self, X, y):

        self.X_train = X_train
        self.y_train = y_train
        
    def predict(self, X_test, k):                                   #Defining predict function
        
        predict = [] 

        for i in range(len(X_test)):                               #For all test data
            distances = []
       
            for x in range(len(X_train)):                         #Calculating distance
                distance = 0
                for y in range(4):
                    distance += pow((X_train[x][y] - X_test[i][y]), 2)
                final =  math.sqrt(distance)
                distances.append(final)
        
            temp = np.argsort(distances)
            neighbors = []
            for x in range(k):                                      #Getting the k-nearest neighbors
                neighbors.append(temp[x])
        

            given_test_class = []                                   #Finding the given class for k-nearest neighbors
            for x in range(len(neighbors)):
                given = y_train[neighbors[x]]
                given_test_class.append(given)      
        
        
            count0 = 0
            count1 = 0
            count2 = 0
            votes = []
            for x in range(len(given_test_class)):                  #Calculating votes
                if given_test_class[x] == 0:
                    count0 = count0 + 1
                elif given_test_class[x] == 1:   
                    count1 = count1 + 1
                elif given_test_class[x] == 2:
                    count2 = count2 + 1;
              
        
            votes.append(count0)
            votes.append(count1)
            votes.append(count2)
                        
   
            prediction = max(votes)
            if count0 == count1 == prediction or count1 == count2 == prediction or count2 == count0 == prediction:
                result = -1
            else:
                
                result = votes.index(prediction)
            predict.append(result)                                 #Predicting the class
        return predict
            
    
    def report(self, X_test, y_test, predict):                      #Calculating accuracy
        correct = 0
        for x in range(len(X_test)):
            if y_test[x] == predict[x]:
                correct = correct + 1       
            accuracy_percent = (correct/len(X_test))*100.0
        return accuracy_percent


def k_validate(accuracy):                                           #Plotting the accuracy against k-values
    k_values = []
    for x in range(0,112):
        k_values.append(x+1)
    
    plt.plot(k_values, accuracy)
    plt.ylabel('Accuracy')
    plt.xlabel("K-values")
    plt.xscale("linear")
    plt.yscale("linear")
    plt.show()
    best = max(accuracy_percentage)                                  #Finding the best k-value
    best2 = accuracy_percentage.index(best)
    print("The best value for K with highest accuracy:",best2 + 1)  #priting the best k-value
    print("The accuracy for k=1:", best,"%")                            #printing the accuracy"


"""End of class"""

"""Main program starts"""

test = KNN()                                                        #Creating a class object

accuracy_percentage = []
runtime = []
for x in range(1,113):                                              #Applying KNN for all possible values of k
    start = time.time()
    var = test.predict(X_test, x)                               
    end = time.time()
    runtime.append(end - start)
    var2 = test.report(X_test, y_test, var)
    accuracy_percentage.append(var2)
    
k_validate(accuracy_percentage)

best_time = min(runtime)                                            #Finding the best k-value
best_time_k = runtime.index(best_time)
print("The minimum running time for predict function is",best_time, "for k-value=",best_time_k)

"""Main program ends"""