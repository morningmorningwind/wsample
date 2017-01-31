from numpy import *
import time
from random import sample as rand_sample
class wsample:
    def __init__(self,D): # D is the dictionary, whose keys are rvs, and values are weights
        self.probs = []
        self.keys = {}
        nk = 0
        for k in D:
            self.probs.append(D[k])
            self.keys[nk] = k
            nk+=1
        self.probs=array(self.probs,dtype=float)
        self.probs=self.probs/self.probs.sum()
        self.setup()
    def setup(self):
        K       = len(self.probs)
        q       = zeros(K)
        J       = zeros(K, dtype=int)
     
        # Sort the data into the outcomes with probabilities
        # that are larger and smaller than 1/K.
        smaller = []
        larger  = []
        for kk, prob in enumerate(self.probs):
            q[kk] = K*prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
     
        # Loop though and create little binary mixtures that
        # appropriately allocate the larger outcomes over the
        # overall uniform mixture.
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
     
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
     
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
     
        self.J=J
        self.q=q
     
    def draw(self):
        K  = len(self.J)
     
        # Draw from the overall uniform mixture.
        kk = int(floor(random.rand()*K))
     
        # Draw from the binary mixture, either keeping the
        # small one, or choosing the associated larger one.
        
        if rand() < self.q[kk]:
            return self.keys[kk]
        else:
            return self.keys[self.J[kk]]
    def spl(self,n,timing=False,nopref=False):
        if timing: tic = time.time()
        # Generate variates.
        if nopref:
            X=rand_sample(self.keys.values(),n)
        else:
            X =[]
            for nn in xrange(n):
                X.append(self.draw())
        
        if timing: print('time elapsed:'+str(time.time()-tic))
        return X
