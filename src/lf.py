import numpy as np
import math
import pickle

class LF:
    '''
    Latent factor model implementation
    '''

    def __init__(self, n=10, learning_rate=0.01, lmbda=0.05, verbose=False):
        '''
        Arguments:
            utilmat: Utility matrix of type <class: UtilMat>
            n: number of latent factors used in the model
            learning_rate: Learning Rate for Stochastic Gradient Descent
            lmbda: Regularisation Coefficient
            iters: Number of iterations
            starting_value: initialisation value for the U and V matrices
        '''
        self.n = n
        self.learning_rate = learning_rate
        self.lmbda = lmbda
        self.verbose = verbose
        # Initialising Latent factor matrices
        self.P = (np.random.random((6040 + 1, self.n)) * 2 - 1) / self.n * 10
        self.Q = (np.random.random((self.n, 3952 + 1)) * 2 - 1) / self.n * 10
        # Stores model history
        self.history = {'train_loss': [], 'val_loss': []}
    
    def train(self, utilmat, iters=10, val_utilmat=None, method='stochastic'):
        '''
        Trains the model using the given method.
        method:
            available:
                stochastic: Uses Stochastic Gradient Descent (SGD) (default)
                als: Alternating Least Squares method
                bias: Considers bias as a learnable parameter and uses SGD
            iters: Number of iterations on the entire dataset
            utilmat: Utility matrix of type <class: UtilMat> contains training data
            val_utilmat: validation data
        '''
        self.utilmat = utilmat
        if method == 'bias':
            return self.train_bias(utilmat, iters, val_utilmat)
        P = self.P
        Q = self.Q
        um = utilmat.um
        # gloabal average rating
        mu = utilmat.mu
        # user bias
        bx = utilmat.bx
        # movie bias
        bi = utilmat.bi
        # Error function:
        # exi = rxi - mu - bx - bi - px.T * qi
        for i in range(iters):
            for user in um:
                for movie in um[user]:
                    # Actual rating
                    rxi = um[user][movie]
                    px = P[user, :].reshape(-1, 1)
                    qi = Q[:, movie].reshape(-1, 1)
                    # Calculate error
                    exi = rxi - mu - bx[user] - bi[movie] - np.dot(px.T, qi)
                    if method == 'als':
                        if i % 2 == 0:
                            px = px + self.learning_rate * (exi * qi - self.lmbda * px)
                        else:
                            qi = qi + self.learning_rate * (exi * px - self.lmbda * qi)
                    else:                
                        # Update parameters
                        px = px + self.learning_rate * (exi * qi - self.lmbda * px)
                        qi = qi + self.learning_rate * (exi * px - self.lmbda * qi)
                    px = px.reshape(-1)
                    qi = qi.reshape(-1)
                    P[user, :] = px
                    Q[:, movie] = qi
            self.P = P
            self.Q = Q
            if self.verbose:
                print('Iteration {}'.format(i+1))
                tloss = self.calc_loss(utilmat)
                print('Training Loss: ', tloss)
                self.history['train_loss'].append(tloss)
                if val_utilmat:
                    vloss = self.calc_loss(val_utilmat)
                    print('Validation Loss: ', vloss)
                    self.history['val_loss'].append(vloss)
        self.P = P
        self.Q = Q
        
    def train_bias(self, utilmat, iters, val_utilmat):
        '''
        Helper function:
            Implements SGD with learnable bias
        '''
        P = self.P
        Q = self.Q
        um = utilmat.um
        # gloabal average rating
        mu = utilmat.mu
        bx = np.random.random(P.shape[0]) * 2 - 1
        bi = np.random.random(Q.shape[1]) * 2 - 1
        # Error function:
        # exi = rxi - mu - bx - bi - px.T * qi
        for i in range(iters):
            for user in um:
                for movie in um[user]:
                    # Actual rating
                    rxi = um[user][movie]
                    px = P[user, :].reshape(-1, 1)
                    qi = Q[:, movie].reshape(-1, 1)
                    # Calculate error
                    exi = rxi - mu - bx[user] - bi[movie] - np.dot(px.T, qi)          
                    # Update parameters
                    px = px + self.learning_rate * (exi * qi - self.lmbda * px)
                    qi = qi + self.learning_rate * (exi * px - self.lmbda * qi)
                    bx[user] += self.learning_rate * (exi - self.lmbda * bx[user])
                    bi[movie] += self.learning_rate * (exi - self.lmbda * bi[movie])
                    px = px.reshape(-1)
                    qi = qi.reshape(-1)
                    P[user, :] = px
                    Q[:, movie] = qi
            # Saving state after each iteration
            self.P = P
            self.Q = Q
            self.bx = bx
            self.bi = bi
            if self.verbose:
                print('Iteration {}'.format(i+1))
                tloss = self.calc_loss(utilmat)
                print('Training Loss: ', tloss)
                self.history['train_loss'].append(tloss)
                if val_utilmat:
                    vloss = self.calc_loss(val_utilmat)
                    print('Validation Loss: ', vloss)
                    self.history['val_loss'].append(vloss)

    def predict(self, user, movie):
        '''
        Finds predicted rating for the user-movie pair
        '''
        mu = self.utilmat.mu
        bx = self.utilmat.bx
        bi = self.utilmat.bi
        # Baseline prediction
        bxi = mu + bx[user] + bi[movie]
        bxi += np.dot(self.P[user, :], self.Q[:, movie])
        return bxi

    def calc_loss(self, utilmat, get_mae=False):
        '''
        Finds the RMSE loss (optional MAE)
        '''
        um = utilmat.um
        mu = utilmat.mu
        bx = utilmat.bx
        bi = utilmat.bi
        cnt = 0
        rmse = 0
        mae = 0
        for user in um:
            for movie in um[user]:
                y = um[user][movie]
                yhat = mu + bx[user] + bi[movie] + np.dot(self.P[user, :], self.Q[:, movie])
                rmse += (y - yhat) ** 2
                mae += abs(y - yhat)
                cnt += 1
        rmse /= cnt
        mae /= cnt
        rmse = math.sqrt(rmse)
        if get_mae:
            return rmse, mae
        return rmse
    
    def save(self, name):
        '''
        Saves the model
        '''
        filename = 'saved/' + name + '.pickle'
        with open(filename, 'wb') as f:
            pickle.dump(self.__dict__, f)

    def load(self, name):
        '''
        Loads the model
        '''
        filename = 'saved/' + name + '.pickle'
        with open(filename, 'rb') as f:
            tmp_dict = pickle.load(f)
        self.__dict__.update(tmp_dict)



