import numpy as np
import logging
from utils import compute_accuracy

#Use L2 distance as the distance metric

class NearestNeighbor(object):
    def __init__(self, data, labels, k):
        """
        Args:
            data: n x d matrix with a d-dimensional feature for each of the n
            points
            labels: n x 1 vector with the label for each of the n points
            k: number of nearest neighbors to use for prediction
        """
        self.k = k
        self.data = data
        self.labels = labels


    def train(self):
        """
        Trains the model and stores in class variables whatever is necessary to
        make predictions later.
        """
        # BEGIN YOUR CODE
        # the nearest neighbor classifier simply remembers all the training data
        pass
        # END YOUR CODE
        
        
    

    def predict(self, x):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
        Returns:
            y: n vector with the predicted label for each of the n points
        """
        
        # BEGIN YOUR CODE
        num_test = x.shape[0]
        #lets make sure that the output type matches the inout type
        Ypred = np.zeros(num_test,dtype = self.labels.dtype)
        #loop over all test rows
        for i in range(num_test):
            #using L2 distance
            #distances =  np.linalg.norm(self.data - x[i,:],axis = 1 )
            distances = np.sqrt(np.sum((self.data - x[i]) ** 2, axis=1)) 
            #for this question, we need to select the number of nearest neighbors k
            # My implementation should be general and handle different values fo k correctly. 
            # Returns the indices that would sort an array.
            knn_indice = np.argsort(distances)[:self.k]
            #extracts the labels of these k nearest training data points
            knn_labels = self.labels[knn_indice]  
            # find the most common label in this case 
            # find each unique elements of an array and count its number
            different_labels, counts = np.unique(knn_labels, return_counts=True)
            # find the max count, the most common k values
            Ypred[i] = different_labels[np.argmax(counts)]
            
        return Ypred
        # END YOUR CODE
        
 


    def get_nearest_neighbors(self, x, k):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
            k: number of nearest neighbors to return
        Returns:
            top_imgs: n x k x d vector containing the nearest neighbors in the
            training data, top_imgs must be sorted by the distance to the
            corresponding point in x.
        """
        
        # BEGIN YOUR CODE
        #Using the L2 distance to compute the distance between train data and each point
        distances = np.empty((x.shape[0], self.data.shape[0]))
        for i in range(x.shape[0]):
            for j in range(self.data.shape[0]):
            #Consider the np.dot function
                a = x[i]
                b = self.data[j]
                distances[i, j] = np.sqrt(np.dot(a, a) + np.dot(b, b) - 2 * np.dot(a, b))
        #find the top k neighbors for each imag (indice)
        indice = np.argsort(distances, axis = 1)[:,:k]
        #extract the nearest image
        top_imgs = self.data[indice]
        return top_imgs
        # END YOUR CODE


class LinearClassifier(object):
    def __init__(self, data, labels, epochs=10, lr=1e-3, reg_wt=3e-5, writer=None):
        self.data = data
        self.labels = labels
        self.epochs = epochs
        self.lr = lr
        self.reg_wt = reg_wt
        self.rng = np.random.RandomState(1234)
        std = 1. / np.sqrt(data.shape[1])
        self.w = self.rng.uniform(-std, std, size=(self.data.shape[1], 10))
        self.writer = writer

    def compute_loss_and_gradient(self):
        """
        Computes total loss and gradient of total loss with respect to weights
        w.  You may want to use the `data, w, labels, reg_wt` attributes in
        this function.
        
        Returns:
            data_loss, reg_loss, total_loss: 3 scalars that represent the
                losses $L_d$, $L_r$ and $L$ as defined in the README.
            grad_w: n x 10. The gradient of the total loss (including
            the regularization term), wrt the weight.
        """
        
        # BEGIN YOUR CODE
        #number of classes
        # c = 10
        #find the number of data points (number of rows)
        numbers = self.data.shape[0]
        #calculate the scores for each data in each class
        score = np.dot(self.data,self.w)
        # for the following part, transform the formula into code
        #the multi-class logistic regression classifier predicts the probability for each class ccc
        # make the score into probabilities 
        e_score = np.exp(score)
        prob = e_score / np.sum(e_score,axis = 1).reshape(-1, 1)
        #using matrix mult to sum the product of the garident
        grad_w = np.dot(self.data.T, prob)
        #initialize the log_sum to 0
        log_sum = 0
        #loop all data points
        for i in range(numbers):
            # considering the data labels
              y = self.labels[i]
            # sum the negative loglikelihood
              log_sum += np.log(prob[i,y])
            # adjusting the gradient for the right label, subtract the feature vector
              grad_w[:,y] -= self.data[i]
        #the gardient of the r_loss and t_loss:
        grad_data = grad_w / numbers
        grad_r = self.reg_wt * self.w
        grad_w = grad_data + grad_r
        #data loss
        data_loss = -1/numbers * log_sum
        # regularization loss
        #a regularization term, LrL_rLr​, is also added to the loss function
        #calculates the sum of the squared weights:sum(self.w * self.w)
        reg_loss = 0.5 * self.reg_wt * np.sum(self.w * self.w)
        #The total loss where lambdaλ is the regularization strength
        # found a value around 1e-4 to work reasonably well
        total_loss = data_loss +reg_loss
        return data_loss, reg_loss, total_loss, grad_w
        # END YOUR CODE

    def train(self):
        """Train the linear classifier using gradient descent"""            
            # BEGIN YOUR CODE
            # You may want to call the `compute_loss_and_gradient` method.
            # You can also print the total loss and accuracy on the training
            # data here for debugging.
        for i in range(self.epochs):
            data_loss, reg_loss, total_loss, grad_w = self.compute_loss_and_gradient()
            #train the linear classifier using gradient descent and update gradient weights
            self.w -= self.lr * grad_w
            
            # END YOUR CODE

    def predict(self, x):
        """
        Args:
            x: n x d matrix with a d-dimensional feature for each of the n
            points
        Returns:
            y: n vector with the predicted label for each of the n points
        """
        score = np.dot(x, self.w)
        prediction = np.argmax(score, axis=1)
        return prediction

