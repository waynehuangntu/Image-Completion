# Image Completion using Statistics of Patch Offsets

import cv2, numpy as np, sys, math, operator, maxflow, random, config as cfg
from scipy import ndimage
from time import time
from itertools import count, combinations
import inpaint_cv as cv


MAX = 100000000.0

class Optimizer(object):
    def __init__(self, image, mask, labels):
        self.image = image/255.0
        self.mask = mask
        self.labels = labels
        x, y = np.where(self.mask != 0)
        #print(x)
        #print(y)
        sites = []
        for (i, j) in zip(x, y):
            sites.append([i, j])
        self.sites = sites
        self.neighbors = []
        self.d = np.zeros((len(sites), len(labels)))
        self.Init_D()
        self.Init_Neighbors()

    def Init_D(self):
        for i in range(len(self.sites)):
            for j in range(len(self.labels)):
                self.d[i,j] = self.D_func(self.sites[i], self.labels[j])
    
    def Init_Neighbors(self):
        start = time()
        for i in range(len(self.sites)):
            tmp = []
            neighbors = self.GetNeighbors(self.sites[i])
            for x in neighbors:
                if x in self.sites:
                    index = self.sites.index(x)
                    tmp.append(index)
            self.neighbors.append(tmp)
        end = time()
        print ("InitializeNeighbors execution time: ", end - start)

    def D_func(self, site, offset):
        i = site[0] + offset[0]
        j = site[1] + offset[1]
        
        try:
            if self.mask[i][j] == 0:
                return 0
            return np.inf
        except:
            return np.inf

    def V_func(self, site1, site2, alpha, beta):
        start = time()
        x1a, y1a = site1[0] + alpha[0], site1[1] + alpha[1]
        x2a, y2a = site2[0] + alpha[0], site2[1] + alpha[1]
        x1b, y1b = site1[0] + beta[0], site1[1] + beta[1]
        x2b, y2b = site2[0] + beta[0], site2[1] + beta[1]
        try:
            if self.mask[x1a, y1a] == 0 and self.mask[x1b, y1b] == 0 and self.mask[x2a, y2a] == 0 and self.mask[x2a, y2a] == 0:
                return np.sum((self.image[x1a, y1a] - self.image[x1b, y1b])**2) + np.sum((self.image[x2a, y2a] - self.image[x2b, y2b])**2)
            return MAX
        except:
            return MAX
        
    def IsLowerEnergy(self, nodes, labelling1, labelling2):
        updatedNodes = np.where(labelling1 != labelling2)[0]
        diff = 0.0
        for node in updatedNodes:
            if self.D_func(self.sites[node], self.labels[labelling2[node]]) < float('inf'):
                for n in self.neighbors[node]:
                    if n in updatedNodes:
                        if n > node:
                            diff += self.V_func(self.sites[node], self.sites[n], self.labels[labelling2[node]], self.labels[labelling2[n]]) - self.V_func(self.sites[node], self.sites[n], self.labels[labelling1[node]], self.labels[labelling1[n]])
                    else:
                        try:
                            diff += self.V_func(self.sites[node], self.sites[n], self.labels[labelling2[node]], self.labels[labelling2[n]]) - self.V_func(self.sites[node], self.sites[n], self.labels[labelling1[node]], self.labels[labelling1[n]])
                        except:
                            print(f"Random permutation causing error, please try it again, it may require multiple times.")
                            exit()
            else:
                return False
        if diff < 0:
            return True
        return False

    def GetNeighbors(self, site):
        up = [site[0]-1, site[1]]
        left = [site[0], site[1] - 1]
        down = [site[0] + 1, site[1]]
        right = [site[0], site[1] + 1]
        return [up, left, down, right] 
        

    def AreNeighbors(self, site1, site2):
        x = np.abs(site1[0]-site2[0])
        y = np.abs(site1[1]-site2[1])
        if x < 2 and y < 2:
            return True
        return False 

    def Init_Labelling(self):
        start = time()
        labelling = [None]*len(self.sites)
        for i in range(len(self.sites)):
            np.random.seed(1000)
            perm = np.random.permutation(len(self.labels))
            #perm = np.random(len)
            #perm = np.array([x for x in range(len(self.labels))])
            for p in perm:
                if self.D_func(self.sites[i], self.labels[p]) < MAX:
                    labelling[i] = p
                    break     
        self.sites = [self.sites[i] for i in range(len(self.sites)) if labelling[i] != None]
        tmp = []
        for label in labelling:
            if label != None:
                tmp.append(label)
        labelling = np.array(tmp)
        end = time()
        print ("InitializeLabelling execution time: ", end - start)
        return self.sites, labelling

    def CreateGraphABS(self, alpha, beta, ps, labelling):
        start = time()
        v = len(ps)
        g = maxflow.Graph[float](v, 3*v)
        nodes = g.add_nodes(v)
        for i in range(len(ps)):
            # add the data terms here
            ta, tb = self.D_func(self.sites[ps[i]], self.labels[alpha]), self.D_func(self.sites[ps[i]], self.labels[beta])
            # add the smoothing terms here
            neighbor_list = self.neighbors[ps[i]]
            for ind in neighbor_list:
                try:
                    a, b, j = labelling[ps[i]], labelling[ind], ps.index(ind)
                    if j > i and (b == alpha or b == beta):
                        epq = self.V_func(self.sites[ps[i]], self.sites[ps[j]], self.labels[alpha], self.labels[beta])
                        g.add_edge(nodes[i], nodes[j], epq, epq)
                    else:
                        ea = self.V_func(self.sites[ps[i]], self.sites[ps[j]], self.labels[alpha], self.labels[b])
                        eb = self.V_func(self.sites[ps[i]], self.sites[ps[j]], self.labels[beta], self.labels[b])
                        ta, tb = ta + ea, tb + eb
                except Exception as e:
                    pass                                  
            g.add_tedge(nodes[i], ta, tb)
        end = time()
        #print "CreateGraph execution time: ", end - start
        return g, nodes

    def CreateGraphAE(self, alpha, labelling):
        start = time()
        v = len(self.sites)
        g = maxflow.Graph[float](2*v, 4*v)
        nodes = g.add_nodes(v)
        for i in range(v):
            ta, tb = self.D(self.sites[i], self.labels[alpha]), float('inf')
            if labelling[i] != alpha:
                tb = self.D(self.sites[i], self.labels[labelling[i]])
            g.add_tedge(nodes[i], ta, tb)
            neighbor_list = self.neighbors[i]
            for j in neighbor_list:
                try:
                    if labelling[i] == labelling[j] and j > i:
                        epq = self.V_func(self.sites[i], self.sites[j], self.labels[labelling[i]], self.labels[alpha])
                        g.add_edge(nodes[i], nodes[j], epq, epq)
                    elif j > i:
                        aux_nodes = g.add_nodes(1)
                        epa = self.V_func(self.sites[i], self.sites[j], self.labels[labelling[i]], self.labels[alpha])
                        eaq = self.V_func(self.sites[i], self.sites[j], self.labels[labelling[j]], self.labels[alpha])
                        epq = self.V_func(self.sites[i], self.sites[j], self.labels[labelling[i]], self.labels[labelling[j]])
                        g.add_edge(nodes[i], aux_nodes[0], epa, epa)
                        g.add_edge(nodes[j], aux_nodes[0], eaq, eaq)
                        g.add_tedge(aux_nodes[0], float('inf'), epq)
                except Exception as e:
                    print(e)
        end = time()
        #print "CreateGraph execution time: ", end - start
        return g, nodes            

    def OptimizeLabellingABS(self, labelling):
        labellings = np.zeros((2, len(self.sites)), dtype=int)
        labellings[0] = labellings[1] = np.copy(labelling)
        iter_count = 0
        while(True):
            start = time()
            success = False
            for alpha, beta in combinations(range(len(self.labels)), 2):
                ps = [i for i in range(len(self.sites)) if (labellings[0][i] == alpha or labellings[0][i] == beta)]
                if len(ps) > 0:
                    g, nodes = self.CreateGraphABS(alpha, beta, ps, labellings[0])
                    flow = g.maxflow()
                    for i in range(len(ps)):
                        gamma = g.get_segment(nodes[i])
                        labellings[1, ps[i]] = alpha*(1-gamma) + beta*gamma
                    if self.IsLowerEnergy(ps, labellings[0], labellings[1]):
                        labellings[0, ps] = labellings[1, ps] 
                        success = True
                    else:
                        labellings[1, ps] = labellings[0, ps]                      
            iter_count += 1
            end = time()
            print ("Iteration " + str(iter_count) + " execution time: ", str(end - start)) 
            if success != True or iter_count >= cfg.MAX_ITER:
                break
        return labellings[0]