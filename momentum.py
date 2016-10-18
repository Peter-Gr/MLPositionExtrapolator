import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1,1,2,2,3],
                [10,1,11,2,12,3,13],  
                [0,0,1,1,2,2,4],
                [10,1,11,2,12,3,14],
                [10,5,13,11,14,13,18]]);

# output dataset            
y = np.array([[3,4,4,5,21]]).T

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((7,1)) - 1

for iter in range(10):
    print ("Run" + str(iter))
    print ("Syn0")
    print (syn0)
    # forward propagation
    l0 = X
    dotProd = np.dot(l0,syn0)
    l1 = nonlin(dotProd)
    print ("l1")
    print (l1)

    # how much did we miss?
    l1_error = y - l1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1,True)

    # update weights
    syn0 += np.dot(l0.T,l1_delta)

print ("Output After Training ")
print (syn0)
print (l1)
print (np.multiply(l1,y))

