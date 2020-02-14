# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 14:36:04 2020

@author: yusuf
"""

def maxDistances(self,edgelength):

    nodes = list(self.G.nodes())
    
    M = 100000
    D = {}
    P = {}
    nodelength = edgelength+1
    start = 0
    end = self.G.number_of_nodes()-1
    D[start,1]=0
    for v in nodes: 
        if v != start: 
            D[v,1]=M

    for k in range(2,nodelength+1):
        for v in nodes:
            candidates = {}
            for u in nodes:
                if (u,v) in self.edges:
                    candidates[u] = D[u,k-1]-self.G[u][v]['weight']
                else:
                    candidates[u] = D[u,k-1]+M
                u_min = min(candidates, key=candidates.get)
                D[v,k] = candidates[u_min]  
                P[v,k] = u_min

    path = []
    v = end
    for k in range(nodelength,1,-1):
        path.append(v)
        v = P[v,k]
    path.append(start)
    path.reverse()

    pathstring = []
    for i in path:
        if i in self.labels:
            pathstring.append(self.labels[i])

    Dmax = {}
    Dmax[1] = -D[self.G.number_of_nodes()-1,1]
    for i in range(2,edgelength+2): Dmax[i] = max(Dmax[i-1],-D[self.G.number_of_nodes()-1,i])

    return Dmax,pathstring