"""

ITM network model

"""
import numpy as np
import node

obsDim = node.obsDim

class Map (object):
    
    def __init__(self):
        self.count = 2
        self.nodes = []
        self.e_max = 6 # the desired mapping resolution
        self.n1, self.n2 = node.Node(), node.Node()
        self.n1.refVec, self.n2.refVec =  np.zeros(obsDim), np.zeros(obsDim)
        self.n1.edges.append(self.n2); self.n2.edges.append(self.n1)              
        self.nodes.append(self.n1); self.nodes.append(self.n2)

        self.lastVisitedNode = 0 # index of last visited node
                               
    def remove_node(self, x):
        for node_ in x.edges:
            node_.edges.remove(x)
            if len(node_.edges)==0:
                self.remove_node(node_)
        self.nodes.remove(x)
        
    def bmn (self, stimulus): # returns best-matching node (bmn)
        distances = np.zeros((len(self.nodes),))
        for i in range(len(self.nodes)):
            distances[i] = np.linalg.norm(self.nodes[i].refVec - stimulus[0:obsDim])
        minimum = distances[0]
        index = 0
        for i in range(len(distances)):
            if distances[i]<minimum:
                minimum = distances[i]
                index = i
        
        return self.nodes[index]
        
    def adapt (self, stimulus, next_r, prev_state_act): # stimulus is the encoded next obs
        lp = self.nodes[self.lastVisitedNode].update_node(prev_state_act, stimulus, next_r)
        
        # matching
        distances = np.zeros((len(self.nodes),))
        for i in range(len(self.nodes)):
            distances[i] = np.linalg.norm(self.nodes[i].refVec - stimulus[0:obsDim])
        minimum = distances[0]
        index1, index2 = 0, 0
        for i in range(len(distances)):
            if distances[i]<minimum:
                minimum = distances[i]
                index1 = i
        minimum = distances[1] if index1==0 else distances[0]
        for i in range(len(distances)):
            if distances[i]<=minimum and i != index1:
                minimum = distances[i]
                index2 = i
        n, s = self.nodes[index1], self.nodes[index2] # nearest & second nearest nodes
        
        # edge adaptation
        if n not in s.edges and s not in n.edges:# and np.dot(np.subtract(n.refVec,stimulus[0:obsDim]),np.subtract(s.refVec,stimulus[0:obsDim]))<0:
            n.edges.append(s)
            s.edges.append(n)
        for node_ in n.edges:
            if  n!=s and s!=node_ and n!=node_ and np.dot(np.subtract(n.refVec,s.refVec),np.subtract(node_.refVec,s.refVec))<0:
                n.edges.remove(node_)
                node_.edges.remove(n)
                if len(node_.edges)==0:
                    self.remove_node(node_)
                    self.count-=1
                    if node_ == self.n1: print ("n1 deleted")
                    
        # node_ adaptation
        self.ok = False
        err_per = np.linalg.norm(stimulus[0:obsDim] - n.refVec)
        if np.dot(np.subtract(n.refVec,stimulus[0:obsDim]),np.subtract(s.refVec,stimulus[0:obsDim]))>0 and \
            err_per > self.e_max:
                self.ok = True
                node_new = node.Node()
                for i in range(obsDim): node_new.refVec[i] = stimulus[i]

                node_new.edges.append(n)
                n.edges.append(node_new)
                self.nodes.append(node_new)  
                self.count+=1
                self.lastVisitedNode = len(self.nodes)-1
        else:
            self.lastVisitedNode = index1
        
        return lp