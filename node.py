from networks import world
import numpy as np
obsDim = 32

class Node(object):
    
    def __init__(self):        
        self.refVec = np.zeros((obsDim,))
        self.edges = [] # list of neighboring nodes
        self.errors = []
        self.avgs = []
        self.max_avg = 1.
        self.avg = 0.
        self.avg_old = -1.
        self.world = world()
        
    def update_node (self, sensorimotor_stimulus, next_s, next_r):
        prediction = self.world.predict(np.array([sensorimotor_stimulus]), batch_size = 1)
        pred_s, pred_r = prediction[0][0], prediction[1][0]
        pred_err = np.linalg.norm(pred_s-next_s) + np.linalg.norm(pred_r-next_r)

        self.errors.append(pred_err)
        if len(self.errors)>40:
            self.errors.pop(0)
        self.avg = np.average(self.errors)
        self.avgs.append(self.avg)
        
        if len(self.avgs)>20:
            self.avgs.pop(0)            
        self.avg_old = 0. if len(self.avgs)==1 else self.avgs[0]
        self.max_avg = max(self.avgs)
        
        lp = self.get_learning_progress()
        
        self.world.fit(x=np.array([sensorimotor_stimulus]), y=[np.array([next_s]), np.array([next_r])], batch_size = 1, epochs = 1, verbose = 0) 
        
        return lp
  
    def get_learning_progress(self):
        normalized_avg, normalized_prev_avg = self.avg/self.max_avg, self.avg_old/self.max_avg
        lp = normalized_prev_avg - normalized_avg
        
        return lp
        
    def get_current_err(self):
        return self.errors[len(self.errors)-1]
        
    def __eq__(self, other):
        if (isinstance(other, self.__class__)):
            return np.array_equal(self.refVec, other.refVec)
    
    def __ne__(self, other):
        if (isinstance(other, self.__class__)):
            return not self.__eq__(other)