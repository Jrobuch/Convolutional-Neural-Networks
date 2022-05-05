# coding: utf-8

# ## Drexel University
# ## CS-615: Deep Learning
# ## HW3
# ## John Obuch

###################################################################################################################################

#### Part 1 ######

print("Part 1: See LaTex PDF document")

###################################################################################################################################

#### Part 2 #####

#import requirements
import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt
from scipy import signal #ok to use pythons conv2 equivelent per profs office hours
import skimage.measure # not allowed to use
from numpy.lib.stride_tricks import as_strided
# %matplotlib inline

print("Part 2:")

#create the two input images
x1 = np.zeros([40,40]) #verticle white line image
x1[0:, 20] = 1 #255 white, 0 black (standardized)
y1 = 0

x2 = np.zeros([40,40]) #horizontal white line image
x2[20,0:] = 1 #255 white, 0 black (standardized)
y2 = 1

#vectorize the images
X = np.array([x1,x2])
Y = np.array([y1,y2])

#initialize theta
np.random.seed(0)
theta = np.reshape(np.random.uniform(-1, 1), (1,1)) 

#initialize the kernel
np.random.seed(0)
K = np.random.uniform(-.0001,.0001, (40,40))

#return/save the initial kernel image
print("Initial Kernel")
print(K)
plt.imsave('InitialKernelP2.png', K, cmap="gray")
# plt.imshow(K, cmap="gray")
# plt.show()

#define convolution function
def get_convolution(X,K):
    return signal.convolve2d(X, K, mode='valid')

#create pooling function that also tracks the index loactions
def pool_wloc(F, kernelSize, s):
    
    '''
    This function does max pooling and also returns the index locations
    F: input matrix
    kernelSize: kernel size
    s: stride
    '''
    outputShape = [((F.shape[0]-kernelSize)//s)+1, ((F.shape[1]-kernelSize)//s)+1] # output shape
    Z = np.empty(outputShape) # create an empty array
    
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    L = np.full((outputShape[0],outputShape[1]), value, dtype=object)
    
    for i in range(0,F.shape[0]-kernelSize+1,s):
        for j in range(0,F.shape[1]-kernelSize+1,s):
            temp = F[i:i+kernelSize, j:j+kernelSize]
            temp_max = np.max(temp)
            temp_max_loc = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
            temp_max_loc = (temp_max_loc[0]+i, temp_max_loc[1]+j)
            
            Z[int(i/s),int(j/s)] = temp_max
            L[int(i/s),int(j/s)] = temp_max_loc
           
    return Z,L

#define the select function
def select(sub, L):
    
    '''
    This function selects the values from the subwindow
    sub: subwindow of the image
    L: index locations
    '''
    output_shape = L.shape
    output = np.empty(output_shape)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            output[i,j] = sub[L[i,j][0], L[i,j][1]]
    return output

#define the linear activation function
def linear(h,theta):
    net = h @ theta
    net = np.array(net)
    return net

#define the cost function
def squared_error(y,y_hat):
    cost = (y-y_hat)**2
    return cost

#establish hyperparameters
iter_thresh = 1000
lr = 0.01
lambda_ = 0.00001
iter_ = 0
RMSE = []

print("Training the data, please wait...")

#train the data
while iter_ < iter_thresh:
    iter_ += 1
    
    #initialize the cost and create empty matricies for gradient updates
    cost = 0
    theta_grad = np.zeros(theta.shape)
    K_grad = np.zeros(K.shape)
    
    #iterate through each image
    for i in range(X.shape[0]):
        
        x,y = X[i], Y[i]
        
        #convolve the image with the kernel
        F = get_convolution(x,K)
        
        #perform max pooling and track the location
        Z, L = pool_wloc(F, 1, 1)
        
        #flatten the pooled window
        h = Z.flatten(order='F')
        
        #compute y_hat
        y_hat = linear(h, theta)

        #incriment the cost
        cost += squared_error(y, y_hat)
        
        #incriment the gradient with respect to theta
        theta_grad += 2*(h.T * (y_hat - y)) + lambda_*theta
        
        #initialize the grdient of K to be a matrix of zeros
        grad = np.zeros(K.shape)
        
        #iterate through each image and update the gradient of the kernel
        for k in range(K.shape[0]):
            for l in range(K.shape[1]):

                H, W = X[i].shape[0], X[i].shape[1]
                M = K.shape[0]
                
                #grab the subwindow
                subwindow = np.fliplr(np.flipud(x[M-k-1:H-k, M-l-1:W-l]))
                
                #select the values based on the max pool locations of the subwindow
                sub_selected = select(subwindow, L)
                
                #flatten the selected values columnwise
                sub_selected_flat = sub_selected.flatten(order='F')
                
                #compute the gradient of each value of the kernel
                grad[k,l] = ((y_hat - y) @ theta.T) @ sub_selected_flat 
                
        #incriment the gradient of K        
        K_grad += grad
    
    #perform the batch of the gradients
    K_grad = K_grad/X.shape[0]
    theta_grad = theta_grad/X.shape[0]
    
    #update theta and kernel paramters
    K = K - lr*K_grad
    theta = theta - lr*theta_grad
    
    #append the Root Mean Squared Error 
    RMSE.append((iter_, np.sqrt(cost/X.shape[0])))

#return the final kernel and display/save the kernel image      
print("Final Kernel")
print(K)
plt.imsave('FinalKernelP2.png', K, cmap="binary")
plt.imshow(K, cmap="binary")
plt.show()

#establish iteration and cost arrays to plot cost vs iteration
epoch = []
cost_ = []
for tup in RMSE:
    epoch.append(tup[0])
    cost_.append(tup[1])
epoch = np.array(epoch)
cost_ = np.array(cost_)

#plot the RMSE vs iteration and return/save the figure
_ = plt.plot(epoch, cost_)
_ = plt.title("RMSE Convergence")
_ = plt.xlabel("Iteration")
_ = plt.ylabel("RMSE")
_ = plt.savefig('CostP2.png')

###############################################################################################################################

###### PART 3: ########

#import requirements
import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt
from scipy import signal #ok to use pythons conv2 equivelent per profs office hours
import skimage.measure # not allowed to use
from numpy.lib.stride_tricks import as_strided
# %matplotlib inline

print("Part 3:")

#create the two input images
x1 = np.zeros([40,40]) #verticle white line image
x1[0:, 20] = 1 #255 white, 0 black (standardized)
y1 = 0

x2 = np.zeros([40,40]) #horizontal white line image
x2[20,0:] = 1 #255 white, 0 black (standardized)
y2 = 1

#vectorize the images
X = np.array([x1,x2])
Y = np.array([y1,y2])

#initialize theta
np.random.seed(0)
theta = np.reshape(np.random.uniform(-0.0001, 0.0001), (1,1))

#initialize the kernel
np.random.seed(0)
K = np.random.uniform(-.0001,.0001, (40,40))

#return/save the initial kernel image
print("Initial Kernel")
print(K)
plt.imsave('InitialKernelP3.png', K, cmap="gray")
# plt.imshow(K, cmap="gray")
# plt.show()

#define the covolution function
def get_convolution(X,K):
    return signal.convolve2d(X, K, mode='valid')

#define the max pool function and keep track of index locations
def pool_wloc(F, kernelSize, s):
    
    '''
    This function does max pooling and also returns the index locations
    F: Input matrix
    kernelSize: kernel size
    s: stride
    '''
    outputShape = [((F.shape[0]-kernelSize)//s)+1, ((F.shape[1]-kernelSize)//s)+1] # output shape
    Z = np.empty(outputShape) # create an empty array
    
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    L = np.full((outputShape[0],outputShape[1]), value, dtype=object)
    
    for i in range(0,F.shape[0]-kernelSize+1,s):
        for j in range(0,F.shape[1]-kernelSize+1,s):
            temp = F[i:i+kernelSize, j:j+kernelSize]
            temp_max = np.max(temp)
            temp_max_loc = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
            temp_max_loc = (temp_max_loc[0]+i, temp_max_loc[1]+j)
            
            Z[int(i/s),int(j/s)] = temp_max
            L[int(i/s),int(j/s)] = temp_max_loc
           
    return Z,L

#define the select function
def select(sub, L):
    
    '''
    This function selects the values from the subwindow
    sub: subimage
    L: locations
    '''
    output_shape = L.shape
    output = np.empty(output_shape)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            output[i,j] = sub[L[i,j][0], L[i,j][1]]
    return output

#define the sigmoid activiation function
def sigmoid(h,theta):
    net = h @ theta
    net = np.array(net)
    net[net<-308] = -308 #avoid overflow issues
    net[net>308] = 308
    return 1/(1+np.exp(-net))

#define the cost function
def log_likely(y,y_hat):
    cost = y*np.log(y_hat + 1e-10) + (1-y)*np.log(1-y_hat + 1e-10)
    return cost

#establish the hyperparameters
iter_thresh = 1000
lr = 0.01
lambda_ = 0.00001
iter_ = 0
COST_L = []

print("Training the data, please wait...")

#train the data
while iter_ < iter_thresh:
    iter_ += 1
    
    #initialize the cost and gradient matricies that empty during each iteration
    cost = 0
    theta_grad = np.zeros(theta.shape)
    K_grad = np.zeros(K.shape)
    
    #iterate through each image
    for i in range(X.shape[0]):
        
        x,y = X[i], Y[i]
        
        #perform the convoluion
        F = get_convolution(x,K)
        
        #perform the max pool and keep track of the index location
        Z, L = pool_wloc(F, 1, 1)
        
        #flatten the max pool
        h = Z.flatten(order='F')
        
        #compute y_hat
        y_hat = sigmoid(h, theta)

        #incriment the cost
        cost += log_likely(y, y_hat)
        
        #incriment the gradient
        theta_grad += h.T * (y - int(y_hat)) - lambda_*theta
        
        #initialize the K gradient matrix of zeros
        grad = np.zeros(K.shape)
        
        #iterate through each image and update the gradient of the kernel
        for k in range(K.shape[0]):
            for l in range(K.shape[1]):

                H, W = X[i].shape[0], X[i].shape[1]
                M = K.shape[0]
                
                #grab the subwindow
                subwindow = np.fliplr(np.flipud(x[M-k-1:H-k, M-l-1:W-l]))
                
                #select the locations of the subwindow
                sub_selected = select(subwindow, L)
                
                #flatten the selected values columnwise
                sub_selected_flat = sub_selected.flatten(order='F')
                
                #update the gradient
                grad[k,l] = ((y-y_hat) @ theta.T) @ sub_selected_flat 
                
        #incriment the gradient
        K_grad += grad
    
    #perform the batch of the gradients by averaging the gradients
    K_grad = K_grad/X.shape[0]
    theta_grad = theta_grad/X.shape[0]
    
    #update the theta and kernel parameters
    K = K + lr*K_grad
    theta = theta + lr*theta_grad
    
    #keep track of the cost for each iteration
    COST_L.append((iter_, np.mean(cost)))

#return/save the final kernel
print("Final Kernel")        
print(K)
plt.imsave('FinalKernelP3.png', K, cmap="gray")
# plt.imshow(K, cmap="gray") #binary
# plt.show()

#create voctors for iteration and cost to plot later
epoch = []
cost_ = []
for tup in COST_L:
    epoch.append(tup[0])
    cost_.append(tup[1])
epoch = np.array(epoch)
cost_ = np.array(cost_)

#return/save the cost vs iteration plot
_ = plt.plot(epoch, cost_)
_ = plt.title("Cost Convergence")
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Cost")
_ = plt.savefig('CostP3.png')
  
###################################################################################################################################

##### PART 4 ######

#import requirements
import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt
from scipy import signal #ok to use pythons conv2 equivelent per profs office hours
import skimage.measure # not allowed to use
from numpy.lib.stride_tricks import as_strided
# %matplotlib inline

print("Part 4:")

#create the two input images
x1 = np.zeros([40,40]) #verticle white line image
x1[0:, 20] = 1 #255 white, 0 black (standardized)
y1 = 0

x2 = np.zeros([40,40]) #horizontal white line image
x2[20,0:] = 1 #255 white, 0 black (standardized)
y2 = 1

#vecotrize the images
X = np.array([x1,x2])
Y = np.array([y1,y2]) 

#one-hot-encode the y-vector
enc = OneHotEncoder(categories='auto')
Y = enc.fit_transform(Y.reshape(-1, 1)).toarray()

#initialize theta parameter
np.random.seed(0)
theta = np.random.uniform(-1, 1,(1,X.shape[0]))

#intialize kernel paramter
np.random.seed(0)
K = np.random.uniform(-.0001,.0001, (40,40))

#return/save kernel image
print("Initial Kernel")
print(K)
plt.imsave('InitialKernelP4.png', K, cmap="gray")
# plt.imshow(K, cmap="gray")
# plt.show()

#define convolution function
def get_convolution(X,K):
    return signal.convolve2d(X, K, mode='valid')

#define max pool function and also keep track of max index locations
def pool_wloc(F, kernelSize, s):
    
    '''
    This function does max pooling and also returns the max value index locations
    F: Input matrix
    kernelSize: Kernel/Width size
    s: Stride
    '''
    outputShape = [((F.shape[0]-kernelSize)//s)+1, ((F.shape[1]-kernelSize)//s)+1] # output shape
    Z = np.empty(outputShape) # create an empty array
    
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    L = np.full((outputShape[0],outputShape[1]), value, dtype=object)
    
    for i in range(0,F.shape[0]-kernelSize+1,s):
        for j in range(0,F.shape[1]-kernelSize+1,s):
            temp = F[i:i+kernelSize, j:j+kernelSize]
            temp_max = np.max(temp)
            temp_max_loc = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
            temp_max_loc = (temp_max_loc[0]+i, temp_max_loc[1]+j)
            
            Z[int(i/s),int(j/s)] = temp_max
            L[int(i/s),int(j/s)] = temp_max_loc
           
    return Z,L

#define the select function
def select(sub, L):
    
    '''
    this function selects the values from sub
    sub: subimage
    L: locations
    '''
    output_shape = L.shape
    output = np.empty(output_shape)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            output[i,j] = sub[L[i,j][0], L[i,j][1]]
    return output

#define the activation function
def softmax(h,theta):
    e_x = np.exp(h @ theta)
    e_x = np.nan_to_num(e_x) #avoid overflow issues as a safeguard
    e_x_sum = np.sum(e_x, axis = 0) #axis = None also works becaue 1-d result after dot product is performed
    return e_x/e_x_sum

#define the cost function
def cross_entropy_loss(y,y_hat):
    cross_loss = -np.sum(y*np.log(y_hat + 1e-10))
    return cross_loss

#establish hyper parameters
iter_thresh = 1000
lr = 0.01
lambda_ = 0.00001
iter_ = 0
COST_L = []

print("Training the data, please wait...")

#train the data
while iter_ < iter_thresh:
    iter_ += 1
    
    #initialize the cost and gradient matrices that are emptied after each iteration 
    cost = 0
    theta_grad = np.zeros(theta.shape)
    K_grad = np.zeros(K.shape)
    F_temp = np.zeros(F.shape)
    
    for i in range(X.shape[0]):
        
        x, y = X[i], Y[i]
        
        #convolve the image X with K
        F = get_convolution(x,K)
        
        #perform max pooling
        Z, L = pool_wloc(F, 1, 1)
        
        #flatten the max pool matrix column-wise
        h = Z.flatten(order='F')
        
        #compute y_hat
        y_hat = softmax(h, theta)
        
        #compute/incriment the cost value
        cost += cross_entropy_loss(y,y_hat)
        
        #incriment/compute the theta gradient
        theta_grad += h.T * (y_hat - y) + lambda_*theta
        
        #initialize an empty matrix to store K values into for each iteration
        grad = np.zeros(K.shape)
        
        #for each element location compute the gradient of K
        for k in range(K.shape[0]):
            for l in range(K.shape[1]):

                H, W = X[i].shape[0], X[i].shape[1]
                M = K.shape[0]
                
                #grab the subwindow
                sub_window = np.fliplr(np.flipud(x[M-k-1:H-k, M-l-1:W-l]))
                
                #select the locations of the subwindow that are associate to the max values of F
                sub_selected = select(sub_window, L)
                
                #flatten the selected values column-wise 
                sub_selected_flat = sub_selected.flatten(order='F')
                
                #compute the gradient of the specific value of K
                grad[k,l] = ((y_hat-y) @ theta.T) @ sub_selected_flat 
                
        #incriment the gradient of K for that value
        K_grad += grad
    
    #perform batch update by averaging the gradients with respect to each image
    K_grad = K_grad/X.shape[0]
    theta_grad = theta_grad/X.shape[0]
    
    #update the kernel and theta parameters
    K = K - lr*K_grad
    theta = theta - lr*theta_grad
    
    #keep track of the iteration vs cost
    COST_L.append((iter_, np.mean(cost)))

#return/save the final kernel image
print("Final Kernel")
print(K)
plt.imsave('FinalKernelP4.png', K, cmap="gray")
# plt.imshow(K, cmap="gray")
# plt.show()

#vectorize the iteration and cost arrays to be able to plot them
epoch = []
cost_ = []
for tup in COST_L:
    epoch.append(tup[0])
    cost_.append(tup[1])
epoch = np.array(epoch)
cost_ = np.array(cost_)

#return/save the iteration vs cost graph
_ = plt.plot(epoch, cost_)
_ = plt.title("Cost Convergence")
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Cost")
_ = plt.savefig('CostP4.png')

###################################################################################################################################

###### PART 5 ###########

#import requirements
import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt
from scipy import signal #ok to use pythons conv2 equivelent per profs office hours
# import skimage.measure # not allowed to use
from numpy.lib.stride_tricks import as_strided
# %matplotlib inline

print("Part 5:")

#create the two input images
x1 = np.zeros([40,40]) #verticle white line image
x1[0:, 20] = 1 #255 white, 0 black (standardized)
y1 = 0

x2 = np.zeros([40,40]) #horizontal white line image
x2[20,0:] = 1 #255 white, 0 black (standardized)
y2 = 1

#vecotrize the images and their associated classes
X = np.array([x1,x2])
Y = np.array([y1,y2]) 

#initialize the theta parameter
np.random.seed(0)
theta = np.random.uniform(-1, 1,(1,324*4)) 

#keep track of the K index/theta location of the flattened h to perform gradient of K update correctly
theta1 = theta[:, :324]
theta2 = theta[:, 324:648]
theta3 = theta[:, 648:972]
theta4 = theta[:, 972:]

#vectorize the above sub theta arrays to be able to iterate through the flattened h for performing the update of the K gradient
Theta = np.array([theta1, theta2, theta3, theta4]) 

#initialize the kernel parameter
np.random.seed(0)
K = np.reshape(np.array([np.random.uniform(-.0001,.0001, (5,5))]*4), (5,5,4)) 

#return/save the intial kernel images
print("Initial Kernels")
for i in range(4):
    print(K[:,:,i])
    plt.imsave('InitialKernel_'+str(i)+'P5.png', K[:,:,i], cmap="gray")
#     plt.imshow(K[:,:,i], cmap="gray")
#     plt.show()

#define the convolution function
def get_convolution(X,K):
    return signal.convolve2d(X, K, mode='valid')

#define the max pool function while also keeping track of the max values index locations
def pool_wloc(F, kernelSize, s):
    
    '''
    This function does max pooling and also returns the associated max value index locations
    F: Input matrix
    kernelSize: Kernel/Width size
    s: Stride
    '''
    outputShape = [((F.shape[0]-kernelSize)//s)+1, ((F.shape[1]-kernelSize)//s)+1] # output shape
    Z = np.empty(outputShape) 
    
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    L = np.full((outputShape[0],outputShape[1]), value, dtype=object)
    
    for i in range(0,F.shape[0]-kernelSize+1,s):
        for j in range(0,F.shape[1]-kernelSize+1,s):
            temp = F[i:i+kernelSize, j:j+kernelSize]
            temp_max = np.max(temp)
            temp_max_loc = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
            temp_max_loc = (temp_max_loc[0]+i, temp_max_loc[1]+j)
            
            Z[int(i/s),int(j/s)] = temp_max
            L[int(i/s),int(j/s)] = temp_max_loc
           
    return Z,L

#define the select function for update of K gradient chaining
def select(sub, L):
    
    '''
    This function selects the values from the subwindow
    sub: Subwindow of the image
    L: Index locations assocated with the max value locations of F
    '''
    output_shape = L.shape
    output = np.empty(output_shape)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            output[i,j] = sub[L[i,j][0], L[i,j][1]]
    return output

#define the activation function
def sigmoid(h,theta):
    net = h @ theta
    net = np.array(net)
    net[net<-308] = -308 #avoding overflow issues
    net[net>308] = 308
    return 1/(1+np.exp(-net))

#define the cost function
def log_likely(y,y_hat):
    cost = y*np.log(y_hat + 1e-10) + (1-y)*np.log(1-y_hat + 1e-10)
    return cost

#establish the hyperparameters
iter_thresh = 1000
lr = 0.01 
lambda_ = 0.00001
iter_ = 0
COST_L = []

print("Training the data, please wait...")

#train the data
while iter_ < iter_thresh:     
    iter_ += 1
    
    #initialize the cost and parameter matricies to be emptied after each iteration
    cost = 0
    theta_grad = np.zeros(theta.shape)
    K_grad = np.zeros(K.shape) 
    F_temp = np.zeros(F.shape)
    
    #for each image
    for i in range(X.shape[0]):
        
        x, y = X[i], Y[i]
        
        #establish empty array to store concatination of h arrays
        h_array = np.array([])
        
        #for each kernel
        for j in range(K.shape[2]): 
            
            kern = K[:,:, j]
            
            #perform the convolution of the image x with kernel k
            F = get_convolution(x,kern)
            
            #perform max pooling with width of 2 and stride of 2
            Z, L = pool_wloc(F, 2, 2)
            
            #flatten the max pool matrix column-wise
            h = np.reshape(Z.flatten(order='F'), (-1,1)) 
            
            #concatinate each flattened h into the h_array defined above
            h_array = np.concatenate([h_array, h], axis = None)
        
        #reshape h array for dimensions to match
        h_array = np.reshape(h_array, (theta.shape[1], 1))
        
        #compute y_hat
        y_hat = sigmoid(h_array.T, theta.T) 
    
        #compute the cost
        cost += log_likely(y,y_hat)
        
        #update theta gradient
        theta_grad += h_array.T * (y-y_hat) - lambda_*theta 
        
        #back propogate K
        for j in range(K.shape[2]):  
            
            kern = K[:,:, j]
            grad = np.zeros(kern.shape)
            
            for k in range(kern.shape[0]):
                for l in range(kern.shape[1]):

                    H, W = X[i].shape[0], X[i].shape[1]
                    M = K.shape[0]
                    
                    #grab the subwindow
                    sub_window = np.fliplr(np.flipud(x[M-k-1:H-k, M-l-1:W-l]))
                    
                    #select the values associated with the max pool value locations of F
                    sub_selected = select(sub_window, L)
                    
                    #flatten the selected values column-wise
                    sub_selected_flat = sub_selected.flatten(order='F')
                    
                    #update the gradient of each value of K
                    grad[k,l] = ((y-y_hat) @ Theta[j]) @ sub_selected_flat 
            
            #incriment the gradient of the j-th kernel K 
            K_grad[:,:,j] += grad
    
    #perform the batch update of the gradient by averaging each of the gradients K and theta
    K_grad = K_grad/X.shape[0]
    theta_grad = theta_grad/X.shape[0]
    
    #update the theta and kernel parameters
    K = K + lr*K_grad
    theta = theta + lr*theta_grad     
    
    #keep track of the iteration vs cost
    COST_L.append((iter_, np.mean(cost)))

#return/save the final trained kernels
print("Final Kernels")
print(K)
for i in range(4):
    plt.imsave('FinalKernel_'+str(i)+'P5.png', K[:,:,i], cmap="gray")
#     plt.imshow(K[:,:,i], cmap="gray")
#     plt.show()

#vectorize the iteration and cost vectors from the tuple list (i.e. the COST_L list)
epoch = []
cost_ = []
for tup in COST_L:
    epoch.append(tup[0])
    cost_.append(tup[1])
epoch = np.array(epoch)
cost_ = np.array(cost_)

#return/save the iteration vs cost graph
_ = plt.plot(epoch, cost_)
_ = plt.title("Cost Convergence")
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Cost")
_ = plt.savefig('CostP5.png')

###################################################################################################################################

###### PART 6 Configuration 1 #######

print("Part 6 Configuration 1:")

import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt
from scipy import signal #ok to use pythons conv2 equivelent per profs office hours
import skimage.measure # not allowed to use
from numpy.lib.stride_tricks import as_strided
# %matplotlib inline


#establish the directory where the files are located
trainDirectory = "yalefaces/"

#store all files in the /yalefaces directory into a list (NOTE: The README.txt file is an element in the list!)
lFileList = []
for fFileObj in os.walk(trainDirectory): 
    lFileList = fFileObj[2]
    break  

#define empty lists to store flattened image row vectors into as well as the class value labels
matrix = []
class_ = []

#obtain X matrix data and Y target vector (i.e. class values)
for file in lFileList:
    
    #if file is not the README.txt file
    if file != 'Readme.txt':
        
        #read in the image, cast to numpy array and resize the image to be dimensions 40x40, and flatten the image
        im = cv2.resize(np.array(Image.open(trainDirectory+file)), (40,40)).flatten()
        
        #reshape the flattened images and append the images to the matrix list
        im = np.reshape(im, (1,im.shape[0]))[0]
        
        #append each row vector to the matrix list
        matrix.append(im)
        
        #split the file and grab first element subject<id>
        subject_id = file.split(".")[0]
        
        #grab the last two values in the string (i.e. the <id> number) and cast it to an integer
        class_val = int(subject_id[-2:])
        
        #append the values to the class_ list
        class_.append(class_val)

#cast the matrix list and class list to a numpy arrays (matrix/vector)
X = np.array(matrix) 
Y= np.array(np.reshape(class_, (len(class_), 1))) 

#concatinate the y vector to the x matrix such that the last column in matrix A is the target values
A = np.concatenate((X, Y), axis=1)

#initialize random number generator
np.random.seed(42) ###SEED OF ZERO WILL CAUSE STANDARD DEVIAITON OF ZERO ERROR USING SEED OF 42!!!

#create list to store sub matricies associated to each class into
class_matrix_L = []

#get the sub-matricies for each class
for c in set(A[:, -1]):
    mask = A[:, -1] == c
    A_c = A[mask]
    class_matrix_L.append(A_c)

#define empty lists to store the rows of train/test sub-matries into to built the training and testing matricies
train_temp = []
test_temp = []

#shuffle the sub-matrices and split them into 2/3 train and 1/3 test split to ensure equal class prior probabilities
for mat in class_matrix_L:
    np.random.shuffle(mat)
    cutoff = math.ceil((2/3)*mat.shape[0])
    train_ = mat[0:cutoff, :]
    test_ = mat[cutoff:, :]
    
    #for each row in the train and test sub-matricies, append each row to the train_temp & test_temp lists
    for row in train_:
        train_temp.append(row)    
    for row in test_:
        test_temp.append(row)
    
#cast the train and test lists to numpy array matricies
A_train = np.array(train_temp) #numpy.ndarray
A_test = np.array(test_temp)

#randomize the rows in the train and test Matrix groups
np.random.shuffle(A_train)
np.random.shuffle(A_test)

#establish train and test groups
X_train = np.array(A_train[:, 0:-1]) #train
Y_train = np.array(A_train[: , -1])
X_test = np.array(A_test[: , 0:-1]) #test
Y_test = np.array(A_test[: , -1])

#one-hot-encode the Y_train and Y_test vectors to be matricies where each column is representative of a class. 
#per Maryam (TA) ok to use sklearn onehot encoder!
enc = OneHotEncoder(categories='auto')
Y_train_onehot = enc.fit_transform(Y_train.reshape(-1, 1)).toarray()
Y_test_onehot = enc.fit_transform(Y_test.reshape(-1, 1)).toarray()

#compute mean and std of training data to identify if there is a zero in the std vector
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)  

####################### CAN USE THIS IF SEED IS 42 BUT NOT IF SEED IS 0 ################

#remove the features that corresponded to std of zero
for indx, i in enumerate(range(len(X_std_train))):
    if X_std_train[i] == 0:
        X_train= np.delete(X_train, indx, axis=1)
        X_test=np.delete(X_test, indx, axis=1)
        
#recompute mean and std of training data
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)

####################################################################

#standardize the data
X_stdz_train = (X_train - X_bar_train)/X_std_train 
X_stdz_test = (X_test - X_bar_train)/X_std_train

###################### CONVOLUTION STEPS ###########################

#store all the images into a list to iterate through and reshape the row vectors to be 40x40 images
image_list_train = []
for row in X_stdz_train:
    image_list_train.append(np.reshape(row, (40,40))) #TRAIN
    
image_list_test = []
for row in X_stdz_test:
    image_list_test.append(np.reshape(row, (40,40))) #TEST

#cast the image list to a numpy array
X_train = np.array(image_list_train)
X_test = np.array(image_list_test)

#Initialize theta
np.random.seed(0)
theta = np.random.uniform(-0.1,0.1,(324,Y_train_onehot.shape[1]))

#Initialize the Kernel
np.random.seed(0)
K = np.random.uniform(-.0001,.0001, (5,5))

#observe the first three standardized images
for i in range(3):
    plt.imsave('FaceImage_'+str(i)+'P3.png', X_train[i], cmap="gray")
#     plt.imshow(X_train[i], cmap="gray")
#     plt.show()

#define the convolution function
def get_convolution(X,K):
    return signal.convolve2d(X, K, mode='valid')

#define the max pool function while also keeping track of the max value index locations
def pool_wloc(F, kernelSize, s):
    
    '''
    This function does max pooling and also returns the index locations of the max values of F 
    F: Input matrix
    kernelSize: Kernel/Width size
    s: Stride
    '''
    outputShape = [((F.shape[0]-kernelSize)//s)+1, ((F.shape[1]-kernelSize)//s)+1] # output shape
    Z = np.empty(outputShape) # create an empty array
    
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    L = np.full((outputShape[0],outputShape[1]), value, dtype=object)
    
    for i in range(0,F.shape[0]-kernelSize+1,s):
        for j in range(0,F.shape[1]-kernelSize+1,s):
            temp = F[i:i+kernelSize, j:j+kernelSize]
            temp_max = np.max(temp)
            temp_max_loc = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
            temp_max_loc = (temp_max_loc[0]+i, temp_max_loc[1]+j)
            
            Z[int(i/s),int(j/s)] = temp_max
            L[int(i/s),int(j/s)] = temp_max_loc
           
    return Z,L

#define the select function for use in updating the gradient of K using chaining
def select(sub, L):
    
    '''
    This function selects the values from the subwindow
    sub: Subwindow
    L: Locations
    '''
    output_shape = L.shape
    output = np.empty(output_shape)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            output[i,j] = sub[L[i,j][0], L[i,j][1]]
    return output

#define the activation function
def sigmoid(h,theta):
    net = h @ theta
    net = np.array(net)
    net[net<-308] = -308 #avoid overflow issues
    net[net>308] = 308
    return 1/(1+np.exp(-net))

#define the cost function
def log_likely(y,y_hat):
    cost = y*np.log(y_hat + 1e-10) + (1-y)*np.log(1-y_hat + 1e-10)
    return cost

#establish the hyperparameters
iter_thresh = 1000
lr = 0.02
lambda_ = 0.00001
iter_ = 0
COST_L = []
COST_TEST_L = []

print("Training the data, please wait...")

print("Attention grader, this will take ~30+ min to train. If you want to wait then great, otherwise see PDF for results.")

#train the data
while iter_ < iter_thresh:
    
    iter_ += 1
    
    #compute the L2 norm
    L2 = np.linalg.norm(theta) 
    
    #initialize the cost and paramter matricies to be emptied after each iteration
    cost = 0
    cost_test = 0
    theta_grad = np.zeros(theta.shape)
    K_grad = np.zeros(K.shape)
    
    #for each image
    for i in range(X_train.shape[0]):
        
        x,y = X_train[i], np.reshape(Y_train_onehot[i], (1, Y_train_onehot[i].shape[0]))  
        
        #perform the convolution of image x with kernel K
        F = get_convolution(x,K)
        
        #perform max pooling with width of 2 and stride of 2 while also keeping track of the location
        Z, L = pool_wloc(F, 2, 2)
        
        #flatten the pooled matrix column-wise
        h = np.reshape(Z.flatten(order='F'), (-1,1))
        
        #compute y_hat for the train data
        y_hat = sigmoid(h.T, theta)
        
        #compute/incriment the cost for the train data
        cost += log_likely(y, y_hat) - lambda_*L2
        
        #compute/incriment the gradient of theta
        theta_grad += h * (y - y_hat) - lambda_*theta  
        
        #intialize the gradient of K to be a matrix of zeros
        grad = np.zeros(K.shape)
        
        #for each locations in the kernel K
        for k in range(K.shape[0]):
            for l in range(K.shape[1]):

                H, W = X_train[i].shape[0], X_train[i].shape[1]
                M = K.shape[0]
                
                #grab the subwindow
                subwindow = np.fliplr(np.flipud(x[M-k-1:H-k, M-l-1:W-l]))
                
                #select the values associated with the max values from F after pooling
                sub_selected = select(subwindow, L)
                
                #flatten the selected values of the matrix column-wise
                sub_selected_flat = sub_selected.flatten(order='F')
                
                #compute the gradient for each element of K
                grad[k,l] = ((y-y_hat) @ theta.T) @ sub_selected_flat 
                
        #incriment the gradient of K
        K_grad += grad
    
    #compute the batch gradient of theta and K by averaging the gradients associated with each image
    K_grad = K_grad/X.shape[0]
    theta_grad = theta_grad/X.shape[0]
    
    #perform the updates of the theat and kernel parameters
    K = K + lr*K_grad
    theta = theta + lr*theta_grad
    
    #keep track of the iteration and cost by appending them as a tuple to the COST_L list
    COST_L.append((iter_, np.mean(cost)))
    
    ### KEEPING TRACK OF TEST COST
    for i in range(X_test.shape[0]):
        
        x_test,y_test = X_test[i], np.reshape(Y_test_onehot[i], (1, Y_test_onehot[i].shape[0]))  
        
        #perform the convolution of x and K for the test data
        F = get_convolution(x_test,K)
        
        #perform max pooling for the test data
        Z, L = pool_wloc(F, 2, 2)
        
        #flatten the resulting pooled matrix
        h = np.reshape(Z.flatten(order='F'), (-1,1))
        
        #compute y_hat for the test data
        y_hat_test = sigmoid(h.T, theta)
        
        #incriment the cost for the test data
        cost_test += log_likely(y_test, y_hat_test) - lambda_*L2
    
    #append the iteration and cost as a tuple to the COST_TEST_L list
    COST_TEST_L.append((iter_, np.mean(cost_test)))
        
#vectorize the cost and iteration by unpacking the tuple cost/iteration to use for plotting the training data   
epoch = []
cost_ = []
for tup in COST_L:
    epoch.append(tup[0])  ##TRAIN DATA
    cost_.append(tup[1])
epoch = np.array(epoch)
cost_ = np.array(cost_)

#plot/save the resulting iteration vs cost graph for the training data
_ = plt.plot(epoch, cost_)
_ = plt.title("Cost Train Convergence")  ##TRAIN DATA
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Cost")
_ = plt.savefig('CostTrainP6.png')
# plt.show()

#vectorize the cost and iteration by unpacking the tuple cost/iteration to use for plotting the testing data  
epoch_test = []
cost_test = []
for tup in COST_TEST_L:
    epoch_test.append(tup[0])  ##TEST DATA
    cost_test.append(tup[1])
epoch_test = np.array(epoch_test)
cost_test = np.array(cost_test)

#plot/save the resulting iteration vs cost graph for the testing data
_ = plt.plot(epoch_test, cost_test)
_ = plt.title("Cost Test Convergence")  ##TEST DATA
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Cost")
_ = plt.savefig('CostTestP6.png')
# plt.show()

#compute the accuracy metrices

#### TRAIN ######

#initialize an empty list to append each Y_hat array into
Y_hat_train = []

for i in range(X_train.shape[0]):
        
        x,y = X_train[i], np.reshape(Y_train_onehot[i], (1, Y_train_onehot[i].shape[0]))  
        
        #perform the convolution with the final trained kernel with each image x
        F = get_convolution(x,K)
        
        #perform max pooling with width of 2 and stride of 2
        Z, L = pool_wloc(F, 2, 2)
        
        #flatten the resulting matrix from the pooling step
        h = np.reshape(Z.flatten(order='F'), (-1,1))
        
        #compute y_hat for the training data
        y_hat_train = sigmoid(h.T, theta)
        
        #append each y_hat to a list
        Y_hat_train.append(y_hat_train[0])

#vecotrize the list of Y_hat_train list to be a matrix
Y_hat_train = np.array(Y_hat_train)

#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_train_pred = np.zeros([Y_hat_train.shape[0], Y_hat_train.shape[1]])  #Train
for i, row in enumerate(Y_hat_train):
    Y_hat_train_pred[i][np.argmax(row)] = 1
    
#confustion matrix train
print("\nTRAIN: CONFUSION MATRIX\n")
conf_train = Y_train_onehot.T @ Y_hat_train_pred    ##As long as you keep track of what axis
print(conf_train)

print("\nTTRAIN ACCURACY:")
acc_train = np.trace(conf_train)/np.sum(conf_train)
print(acc_train)

####### TEST ########

#initialize an empty list to append each Y_hat array into
Y_hat_test = []

for i in range(X_test.shape[0]):
        
        x,y = X_test[i], np.reshape(Y_test_onehot[i], (1, Y_test_onehot[i].shape[0]))  
        
        #perform max the convolution with the trained Kernel K
        F = get_convolution(x,K)
        
        #perform max pooling with width of 2 and stride of 2
        Z, L = pool_wloc(F, 2, 2)
        
        #flatten the resulting matrix from max pooling
        h = np.reshape(Z.flatten(order='F'), (-1,1))
        
        #compute y_hat for the testing data
        y_hat_test = sigmoid(h.T, theta)
        
        #append each y_hat to the list defined above
        Y_hat_test.append(y_hat_test[0])

#vectorize the list of y_hat arrays to be a matrix
Y_hat_test = np.array(Y_hat_test)

#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_test_pred = np.zeros([Y_hat_test.shape[0], Y_hat_test.shape[1]])  #Test
for i, row in enumerate(Y_hat_test):
    Y_hat_test_pred[i][np.argmax(row)] = 1
    
#confustion matrix test
print("\nTEST: CONFUSION MATRIX\n")
conf_test = Y_test_onehot.T @ Y_hat_test_pred    ##As long as you keep track of what axis
print(conf_test)

print("\nTEST ACCURACY:")
acc_test = np.trace(conf_test)/np.sum(conf_test)
print(acc_test)

###############################################################################################################################

#### Part 6 Configuration 2 #######

print("Part 6 Configuration 2:")

import math, os, cv2, random, collections, importlib
import numpy as np
from PIL import Image
from sklearn.preprocessing import OneHotEncoder #ok to use per Maryam (TA)
from collections import Counter
from matplotlib import pyplot as plt
from scipy import signal #ok to use pythons conv2 equivelent per profs office hours
import skimage.measure # not allowed to use
from numpy.lib.stride_tricks import as_strided
# %matplotlib inline


#establish the directory where the files are located
trainDirectory = "yalefaces/"

#store all files in the /yalefaces directory into a list (NOTE: The README.txt file is an element in the list!)
lFileList = []
for fFileObj in os.walk(trainDirectory): 
    lFileList = fFileObj[2]
    break  

#define empty lists to store flattened image row vectors into as well as the class value labels
matrix = []
class_ = []

#obtain X matrix data and Y target vector (i.e. class values)
for file in lFileList:
    
    #if file is not the README.txt file
    if file != 'Readme.txt':
        
        #read in the image, cast to numpy array and resize the image to be dimensions 40x40, and flatten the image
        im = cv2.resize(np.array(Image.open(trainDirectory+file)), (40,40)).flatten()
        
        #reshape the flattened images and append the images to the matrix list
        im = np.reshape(im, (1,im.shape[0]))[0]
        
        #append each row vector to the matrix list
        matrix.append(im)
        
        #split the file and grab first element subject<id>
        subject_id = file.split(".")[0]
        
        #grab the last two values in the string (i.e. the <id> number) and cast it to an integer
        class_val = int(subject_id[-2:])
        
        #append the values to the class_ list
        class_.append(class_val)

#cast the matrix list and class list to a numpy arrays (matrix/vector)
X = np.array(matrix) 
Y= np.array(np.reshape(class_, (len(class_), 1))) 

#concatinate the y vector to the x matrix such that the last column in matrix A is the target values
A = np.concatenate((X, Y), axis=1)

#initialize random number generator
np.random.seed(42)   ####Maryam says to average the accuracies without random seed works with seed of 42!!!!

#create list to store sub matricies associated to each class into
class_matrix_L = []

#get the sub-matricies for each class
for c in set(A[:, -1]):
    mask = A[:, -1] == c
    A_c = A[mask]
    class_matrix_L.append(A_c)

#define empty lists to store the rows of train/test sub-matries into to built the training and testing matricies
train_temp = []
test_temp = []

#shuffle the sub-matrices and split them into 2/3 train and 1/3 test split to ensure equal class prior probabilities
for mat in class_matrix_L:
    np.random.shuffle(mat)
    cutoff = math.ceil((2/3)*mat.shape[0])
    train_ = mat[0:cutoff, :]
    test_ = mat[cutoff:, :]
    
    #for each row in the train and test sub-matricies, append each row to the train_temp & test_temp lists
    for row in train_:
        train_temp.append(row)    
    for row in test_:
        test_temp.append(row)
    
#cast the train and test lists to numpy array matricies
A_train = np.array(train_temp) #numpy.ndarray
A_test = np.array(test_temp)

#randomize the rows in the train and test Matrix groups
np.random.shuffle(A_train)
np.random.shuffle(A_test)

#establish train and test groups
X_train = np.array(A_train[:, 0:-1]) #train
Y_train = np.array(A_train[: , -1])
X_test = np.array(A_test[: , 0:-1]) #test
Y_test = np.array(A_test[: , -1])

#one-hot-encode the Y_train and Y_test vectors to be matricies where each column is representative of a class. 
#per Maryam (TA) ok to use sklearn onehot encoder!
enc = OneHotEncoder(categories='auto')
Y_train_onehot = enc.fit_transform(Y_train.reshape(-1, 1)).toarray()
Y_test_onehot = enc.fit_transform(Y_test.reshape(-1, 1)).toarray()

#compute mean and std of training data to identify if there is a zero in the std vector
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)  

#remove the features that corresponded to std of zero
for indx, i in enumerate(range(len(X_std_train))):
    if X_std_train[i] == 0:
        X_train= np.delete(X_train, indx, axis=1)
        X_test=np.delete(X_test, indx, axis=1)
        
#recompute mean and std of training data
X_bar_train = np.mean(X_train, axis = 0)
X_std_train = np.std(X_train, axis = 0, ddof = 1)

#standardize the data
X_stdz_train = (X_train - X_bar_train)/X_std_train 
X_stdz_test = (X_test - X_bar_train)/X_std_train

###################### CONVOLUTION STEPS ###########################

#store all the images into a list to iterate through
image_list_train = []
for row in X_stdz_train:
    image_list_train.append(np.reshape(row, (40,40))) #TRAIN
    
image_list_test = []
for row in X_stdz_test:
    image_list_test.append(np.reshape(row, (40,40))) #TEST

#cast the image list to a numpy arrays
X_train = np.array(image_list_train)
X_test = np.array(image_list_test)

#Initialize theta
np.random.seed(42)
theta = np.random.uniform(-.1,.1,(324,Y_train_onehot.shape[1])) #WAS -1,1

#Initialize the Kernel
np.random.seed(42)
K = np.random.uniform(-.0001,.0001, (5,5))

#observe the first three standardized images
# for i in range(3):
#     plt.imshow(X_train[i], cmap="gray")
#     plt.show()

#define the convolution function
def get_convolution(X,K):
    return signal.convolve2d(X, K, mode='valid')

#define the max pool function while also keeping track of the max value index locations
def pool_wloc(F, kernelSize, s):
    
    '''
    This function does max pooling and also returns the locations of the max values associated with F
    F: Input matrix
    kernelSize: Kernel/Width size
    s: Stride
    '''
    outputShape = [((F.shape[0]-kernelSize)//s)+1, ((F.shape[1]-kernelSize)//s)+1] # output shape
    Z = np.empty(outputShape) # create an empty array
    
    value = np.empty((), dtype=object)
    value[()] = (0, 0)
    L = np.full((outputShape[0],outputShape[1]), value, dtype=object)
    
    for i in range(0,F.shape[0]-kernelSize+1,s):
        for j in range(0,F.shape[1]-kernelSize+1,s):
            temp = F[i:i+kernelSize, j:j+kernelSize]
            temp_max = np.max(temp)
            temp_max_loc = np.unravel_index(np.argmax(temp, axis=None), temp.shape)
            temp_max_loc = (temp_max_loc[0]+i, temp_max_loc[1]+j)
            
            Z[int(i/s),int(j/s)] = temp_max
            L[int(i/s),int(j/s)] = temp_max_loc
           
    return Z,L

#define the select function
def select(sub, L):
    
    '''
    this function selects the values from the subwindow
    sub: subimage
    L: locations
    '''
    output_shape = L.shape
    output = np.empty(output_shape)
    for i in range(L.shape[0]):
        for j in range(L.shape[1]):
            output[i,j] = sub[L[i,j][0], L[i,j][1]]
    return output

#define the activation function
def softmax(h,theta):
    e_x = np.exp(h @ theta)
#     e_x = np.nan_to_num(e_x)
    e_x_sum = np.sum(e_x)
    return e_x/e_x_sum

#define the cost function
def cross_entropy_loss(y,y_hat):
    cross_loss = -np.sum(y*np.log(y_hat + 1e-10))
#     cross_loss = -np.log(y_hat + 1e-10)  ##IS THIS CORRECT?
    return cross_loss

#establish the hyperparameters
iter_thresh = 1000
lr = 0.02 ## WAS 0.01
lambda_ = 0.00001
iter_ = 0
COST_L = []
COST_TEST_L = []

print("Training the data, please wait...")

print("Attention grader, this will take ~30+ min to train. If you want to wait then great, otherwise see PDF for results.")

#train the data
while iter_ < iter_thresh:
    
    iter_ += 1
    
    #compute the L2 norm for the cost function
    L2 = np.linalg.norm(theta) 
    
    #initialize the costs and parameter matricies to be emptied after each iteration
    cost = 0
    cost_test = 0
    theta_grad = np.zeros(theta.shape)
    K_grad = np.zeros(K.shape)
    
    #for each image
    for i in range(X_train.shape[0]):
        
        x,y = X_train[i], np.reshape(Y_train_onehot[i], (1, Y_train_onehot[i].shape[0]))  
        
        #perform the convolution
        F = get_convolution(x,K)
        
        #perform max pooling and keep track of locations
        Z, L = pool_wloc(F, 2, 2)
        
        #flatten the resulting max pool matrix
        h = np.reshape(Z.flatten(order='F'), (-1,1))
        
        #compute y_hat
        y_hat = softmax(h.T, theta)  
        
        #compute/incriment the cost for the trianing data
        cost += cross_entropy_loss(y, y_hat) + lambda_*L2
        
        #compute the gradient of theta
        theta_grad += h * (y_hat-y) + lambda_*theta  
        
        #initialize an empty matrix of zeros for K gradient
        grad = np.zeros(K.shape)
        
        #for each element/index location in the kernel
        for k in range(K.shape[0]):
            for l in range(K.shape[1]):

                H, W = X_train[i].shape[0], X_train[i].shape[1]
                M = K.shape[0]
                
                #grab the subwindow
                subwindow = np.fliplr(np.flipud(x[M-k-1:H-k, M-l-1:W-l]))
                
                #select the locations of the subwindow associated with the max value locations of F
                sub_selected = select(subwindow, L)
                
                #flatten the resulting selected values of the matrix column-wise
                sub_selected_flat = sub_selected.flatten(order='F')
                
                #compute the gradient for each element location in the kernel K
                grad[k,l] = ((y_hat-y) @ theta.T) @ sub_selected_flat 
        
        #incriment the kernel gradient
        K_grad += grad
    
    #perform the batch of the gradient updates by averaging the gradients associated to each image
    K_grad = K_grad/X.shape[0]
    theta_grad = theta_grad/X.shape[0]
    
    #perform the update of the theta and Kernel K parameters
    K = K - lr*K_grad
    theta = theta - lr*theta_grad
    
    #keep track of the iteration vs cost and append them to the COST_L list for each iteration
    COST_L.append((iter_, np.mean(cost)))
    
    ### KEEPING TRACK OF TEST COST
    for i in range(X_test.shape[0]):
        
        x_test,y_test = X_test[i], np.reshape(Y_test_onehot[i], (1, Y_test_onehot[i].shape[0]))  
        
        #perform the convoluions with the test data
        F = get_convolution(x_test,K)
        
        #perform max pooling of the test data
        Z, L = pool_wloc(F, 2, 2)
        
        #flatten the max pool matrix
        h = np.reshape(Z.flatten(order='F'), (-1,1))
        
        #compute y_hat for the test data
        y_hat_test = softmax(h.T, theta)
        
        #keep track of the cost of the test data
        cost_test += cross_entropy_loss(y_test, y_hat_test) #- lambda_*L2
    
    #append the iteration and cost for the test data to the COST_TEST_L list
    COST_TEST_L.append((iter_, np.mean(cost_test)))
        
#vectorize the cost and iteration by unpacking the tuple cost/iteration to use for plotting the training data  
epoch = []
cost_ = []
for tup in COST_L:
    epoch.append(tup[0])  ##TRAIN DATA
    cost_.append(tup[1])
epoch = np.array(epoch)
cost_ = np.array(cost_)

#plot/save the resulting iteration vs cost graph for the training data
_ = plt.plot(epoch, cost_)
_ = plt.title("Cost Train Convergence")  ##TRAIN DATA
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Cost")
_ = plt.savefig('CostTrainP6config2.png')
# plt.show()

#vectorize the cost and iteration by unpacking the tuple cost/iteration to use for plotting the testing data
epoch_test = []
cost_test = []
for tup in COST_TEST_L:
    epoch_test.append(tup[0])  ##TEST DATA
    cost_test.append(tup[1])
epoch_test = np.array(epoch_test)
cost_test = np.array(cost_test)

#plot/save the resulting iteration vs cost graph for the testing data
_ = plt.plot(epoch_test, cost_test)
_ = plt.title("Cost Test Convergence")  ##TEST DATA
_ = plt.xlabel("Iteration")
_ = plt.ylabel("Cost")
_ = plt.savefig('CostTestP6config2.png')
# plt.show()

#compute the accuracy metrics

#### TRAIN ######

#initilzie a list to append each y_hat array into 
Y_hat_train = []

for i in range(X_train.shape[0]):
        
        x,y = X_train[i], np.reshape(Y_train_onehot[i], (1, Y_train_onehot[i].shape[0]))  
        
        #perform the convolution with the trained kernel K for each image x
        F = get_convolution(x,K)
        
        #perform max pooling with width of 2 and stride of 2 for the training data
        Z, L = pool_wloc(F, 2, 2)
        
        #flatten the resulting matrix from the pooling step
        h = np.reshape(Z.flatten(order='F'), (-1,1))
        
        #ccompute y_hat 
        y_hat_train = softmax(h.T, theta)
        
        #append each y_hat outcome to the list defined above
        Y_hat_train.append(y_hat_train[0])

#vecotrize the list of y_hat arrays to create a matrix
Y_hat_train = np.array(Y_hat_train)

#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_train_pred = np.zeros([Y_hat_train.shape[0], Y_hat_train.shape[1]])  #Test
for i, row in enumerate(Y_hat_train):
    Y_hat_train_pred[i][np.argmax(row)] = 1
    
#confustion matrix test
print("\nTRAIN: CONFUSION MATRIX\n")
conf_train = Y_train_onehot.T @ Y_hat_train_pred    ##As long as you keep track of what axis
print(conf_train)

#compute the accuracy of the system
print("\nTRAIN ACCURACY:")
acc_train = np.trace(conf_train)/np.sum(conf_train)
print(acc_train)

####### TEST ########

#initialize an empty list to store each y_hat array into
Y_hat_test = []

for i in range(X_test.shape[0]):
        
        x,y = X_test[i], np.reshape(Y_test_onehot[i], (1, Y_test_onehot[i].shape[0]))  
        
        #perform the convolution of each test image x with the trained kernel K
        F = get_convolution(x,K)
        
        #perform max pooling with width of 2 and stride of 2
        Z, L = pool_wloc(F, 2, 2)
        
        #flatten the resulting pooling matrix column-wise
        h = np.reshape(Z.flatten(order='F'), (-1,1))
        
        #compute y_hat
        y_hat_test = softmax(h.T, theta)
        
        #append the resulting y_hat array to the list defined above
        Y_hat_test.append(y_hat_test[0])

#vecorize the Y_hat_test list to be a matrix
Y_hat_test = np.array(Y_hat_test)

#creating an empty Y_train_prediction matrix where each elelement is zero
Y_hat_test_pred = np.zeros([Y_hat_test.shape[0], Y_hat_test.shape[1]])  #Test
for i, row in enumerate(Y_hat_test):
    Y_hat_test_pred[i][np.argmax(row)] = 1
    
#confustion matrix test
print("\nTEST: CONFUSION MATRIX\n")
conf_test = Y_test_onehot.T @ Y_hat_test_pred    ##As long as you keep track of what axis
print(conf_test)

#compute the accuracy of the test system
print("\nTEST ACCURACY:")
acc_test = np.trace(conf_test)/np.sum(conf_test)
print(acc_test)
