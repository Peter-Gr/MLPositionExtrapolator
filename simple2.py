import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
# The pattern is - doulbe the first column other columns are irrelevant
X = np.array([  [2,3,4],
                [3,2,60],
                [4,9,12],
                [5,5,332] ])
    
# output dataset            
y = np.array([[4,6,8,10]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1

for iter in range(200000):
    # forward propagation
    l0 = X
    dotProd = np.dot(l0,syn0)
    l1 = nonlin(dotProd)

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print ("Output After Training ")
print ("L1")
print (l1)
print ("Last Syn0")
print (syn0)
print ("Last Dot Product")
print (dotProd)
print ("Prediction - hoping for 12")
l0 = np.array([[6,12,57]])
dotProd = np.dot(l0,syn0)
l1 = nonlin(dotProd)
print ("L1")
print (l1)
print ("Dot Product")
print (dotProd)


