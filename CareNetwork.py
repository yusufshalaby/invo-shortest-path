import networkx as nx
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt

from utils import AbsoluteDualityGap, optimizeFOPdual, vectorizePaths

class CareNetwork:
    
    def __init__(self):
        self.G = nx.DiGraph()
        self.pos = []
        self.step_index = {}
        self.labels = {}
        self.edges = []
        self.c = []
        self.y = []
        self.__networkFlag = False
        self.__cFlag = False
    
    def formNetwork(self, steps, end_steps):
    
        N1 = len(steps)
        N2 = len(end_steps)

        self.step_index = dict({step:[steps.index(step)+1,steps.index(step)+1+N1] for step in steps},
                        **{step:[2*N1+1+list(end_steps.keys()).index(step),
                                 2*N1+1+list(end_steps.keys()).index(step)+len(end_steps)] for step in end_steps.keys()})
        self.step_index['START'] = [0]
        self.step_index['END'] = [2*N1+2*N2+1]

        self.labels = {self.step_index[step][0]:step for step in self.step_index.keys()}

        G0 = nx.complete_graph(N1)
        pos1 = nx.circular_layout(G0)
        pos2 = {}
        for p in pos1:
            pos2[p]=[pos1[p][0]-0.2, pos1[p][1]]

        flipped_pos1 = {node: (-x,y) for (node, (x,y)) in pos1.items()}
        flipped_pos2 = {node: (-x,y) for (node, (x,y)) in pos2.items()}
        pos1 = list(flipped_pos1.values())
        pos2 = list(flipped_pos2.values())

        self.pos = [(-1.4,0)] + pos1 + pos2 + list(zip(1.7*np.ones(N2),np.linspace(-1,1,N2))) + list(zip(1.9*np.ones(N2),
                                                                                                   np.linspace(-1,1,N2))) +[(2.1,0)]

        self.G.add_nodes_from(range(2*N1+2*N2+2))
        for step in steps:
            self.G.add_edge(self.step_index['START'][0],self.step_index[step][0])
            self.G.add_edge(self.step_index[step][0],self.step_index[step][-1])
            for step2 in steps:
                self.G.add_edge(self.step_index[step][-1],self.step_index[step2][0])
            if len(end_steps)==0:
                self.G.add_edge(self.step_index[step][-1],self.step_index['END'][0])
        for end_step in end_steps.keys():
            for step in end_steps[end_step]:
                self.G.add_edge(self.step_index[step][1],self.step_index[end_step][0])
            self.G.add_edge(self.step_index[end_step][0],self.step_index[end_step][1])
            self.G.add_edge(self.step_index[end_step][1],self.step_index['END'][0])
        self.edges = list(self.G.edges())
        
        self.__N = len(self.step_index.keys())-2
        self.__A = nx.incidence_matrix(self.G, oriented=True).toarray()
        self.__b = np.zeros(self.G.number_of_nodes())
        self.__b[0] = -1
        self.__b[self.G.number_of_nodes()-1] = 1
        self.__networkFlag = True
        
    def __missingNetworkError(self):
        print('Network does not exist')
            
    def setEdgeWeights(self, c, y = []):
        if self.__networkFlag:
            if (len(c) == self.G.number_of_edges()):
                for e in self.edges:
                    self.G[e[0]][e[1]]['weight'] = c[self.edges.index(e)]
                self.c = c
                if len(y) == self.G.number_of_nodes():
                    self.y = y
                else:
                    _,_,_, self.y = optimizeFOPdual(self.G,c)
                self.__cFlag = True
            else: print('Wrong cost vector dimensions')
        else: self.__missingNetworkError()
        
    def showNetwork(self,show_labels=True,show_weights=False,path=None):
        if self.__networkFlag:
            fig = plt.figure(figsize=(16,8))
            ax = fig.add_subplot(111, aspect='equal')
            
            n = self.G.number_of_nodes()
            visibleNodes = self.G.nodes()

            # draw the regular interior nodes in the graph
            nx.draw_networkx_nodes(self.G,self.pos,nodelist=visibleNodes,node_color='grey',node_size=100,ax=ax, alpha=0.2)

            # draw the origin and destination nodes
            for nodes, color in zip([[0],[n-1]],['g','r']):
                for color2, alpha in zip(['w',color],[1,.2]):
                    nx.draw_networkx_nodes(self.G,self.pos,
                                   nodelist=nodes,
                                   node_color=color2,
                                   node_size=200,
                                   ax=ax,alpha=alpha)


            if path is None:
                alpha = 1.0
            else:
                alpha = .15

            edge2ind = {e:i for i,e in enumerate(self.G.edges())}
            ind2edge = {i:e for i,e in enumerate(self.G.edges())}

            visibleEdges = [i for i in range(len(edge2ind)) if ind2edge[i][0] in visibleNodes and ind2edge[i][1] in visibleNodes]

            edgelist = [ind2edge[i] for i in visibleEdges]

            if (show_weights) & (self.__cFlag):
                p = [self.c[i] for i in visibleEdges]  
            else:
                print('Edge weights not shown or undefined')
                show_weights = False
                p = [20.]*self.G.number_of_edges()
                alpha = 0.15

            p = [p[i] for i in visibleEdges]

            # draw edges of graph, make transparent if we're drawing a path over them
            if show_weights:
                edges = nx.draw_networkx_edges(self.G,self.pos,edge_color=p,width=1,
                                           edge_cmap=plt.cm.RdYlBu_r,arrows=False,edgelist=edgelist,edge_vmin=min(p),
                                           edge_vmax=max(p),ax=ax,alpha=alpha)
            else:
                edges = nx.draw_networkx_edges(self.G,self.pos,edge_color=p,width=1,
                                               edge_cmap=plt.cm.binary,arrows=False,edgelist=edgelist,edge_vmin=1,
                                               edge_vmax=20,ax=ax,alpha=alpha)


            # draw the path, only between visible nodes
            if path is not None:
                path = np.flatnonzero(vectorizePaths(Prob.step_index,[path],Prob.edges)[0])
                visiblePath = [i for i in path if ind2edge[i][0] in visibleNodes and ind2edge[i][1] in visibleNodes]
                path_pairs = [ind2edge[i] for i in visiblePath]
                path_colors = [p[i] for i in visiblePath]
                if show_weights:
                    edges = nx.draw_networkx_edges(self.G,self.pos,edge_color=path_colors,width=1,
                                               edge_cmap=plt.cm.RdYlBu,edgelist=path_pairs,arrows=False,edge_vmin=min(p),
                                           edge_vmax=max(p))
                else:
                    edges = nx.draw_networkx_edges(self.G,self.pos,edge_color=path_colors,width=1,
                                                   edge_cmap=plt.cm.binary,edgelist=path_pairs,arrows=True,edge_vmin=1,
                                                   edge_vmax=20)             

            if show_weights:
                cbar = fig.colorbar(edges)
                cbar.set_label(r'    $c_{ij}$',rotation=0,fontsize=18)

            if show_labels:
                labelspos = []
                for i in range(len(self.pos)):
                    if (i>0) & (i<len(self.pos)-1):
                        labelspos.append([self.pos[i][0]+0.1,self.pos[i][1]+0.05])
                    else:
                        labelspos.append(self.pos[i])
                nx.draw_networkx_labels(self.G,labelspos,self.labels,font_weight = 'bold', font_size=12)

            ax.axis([-2.2,2.2,-1.2,1.2])
            #ax.axis('tight')
            #ax.axis('equal')
            ax.axis('off')

        else: self.__missingNetworkError()
    
    def getCoefficients(self, refpaths, goodpaths = [], badpaths = [], step_ranks = [], 
                        subpath_constraints = [], penalty_constraints = [], normIndex = None, exp=2, solver = 'GUROBI',reg_par=0):
      
        if self.__networkFlag:

            constraints = []
            capconstraints = []
            naivecdict = {}

            refpoints = vectorizePaths(self.step_index, refpaths, self.edges) 
            goodpoints = vectorizePaths(self.step_index, goodpaths, self.edges)
            badpoints = vectorizePaths(self.step_index, badpaths, self.edges)

            edgesdf = pd.DataFrame(self.edges,columns=['source','sink'])
            for i in range(len(step_ranks)-1):
                constraint = []
                step1_enter = self.step_index[step_ranks[i]][0]
                step1_exit = self.step_index[step_ranks[i]][-1]
                indices1 = list(edgesdf[edgesdf.source.isin([step1_enter,step1_exit])|edgesdf.sink.isin([step1_enter,step1_exit])].index)
                constraint.append(indices1)

                step2_enter = self.step_index[step_ranks[i+1]][0]
                step2_exit = self.step_index[step_ranks[i+1]][-1]
                indices2 = list(edgesdf[edgesdf.source.isin([step2_enter,step2_exit])|edgesdf.sink.isin([step2_enter,step2_exit])].index)
                constraint.append(indices2)

                constraints.append(constraint)

            for subpath_constraint in subpath_constraints:
                constraint = []
                for component in subpath_constraint:
                    indices = list(np.flatnonzero(vectorizePath(self.step_index,[component],self.edges)[0]>0))
                    constraint.append(indices)
                constraints.append(constraint)

            for penalty_constraint in penalty_constraints:
                constraint = []
                indices = list(np.flatnonzero(vectorizePaths(self.step_index,[penalty_constraint[0]],self.edges)[0]>0))
                constraint.append(indices)
                constraint.append(penalty_constraint[1])
                capconstraints.append(constraint)     


            c, y = AbsoluteDualityGap(A=self.__A, b=self.__b, refpoints=refpoints, goodpoints = goodpoints, 
                                            badpoints = badpoints, solver=solver, exp=exp, constraints = constraints, 
                                            capconstraints = capconstraints, normIndex = normIndex,reg_par=reg_par)

            c = np.asarray(c)
            y = np.asarray(y)

            self.setEdgeWeights(c, y)
        
        else: self.__missingNetworkError()
    
    def getEpsilon(self, paths, exp=1):
        
        if self.__cFlag:
            obj = self.y[-1]-self.y[0]
            points = vectorizePaths(self.step_index, paths, self.edges)
            epsilon = (np.dot(points, self.c) - obj)**exp
            return epsilon
        else: 
            print('Edge weights undefined')
    
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

    def getOmega(self, paths, exp = 1):
        
        if self.__cFlag: 
            obj = self.y[-1]-self.y[0]

            pathlengths = np.sum(vectorizePaths(self.step_index, paths, self.edges),axis=1)
            maxlength = np.int(max(pathlengths))
            Dmax, worstpath = self.maxDistances(maxlength)

            #points = vectorizePaths(self.step_index,paths,self.edges)
            #numerator = (np.dot(points,self.c)-obj)**exp
            numerator = self.getEpsilon(paths,exp)
            denom = (np.array([Dmax[length+1]-obj for length in pathlengths]))**exp
            omega = 1-numerator/denom
        else:
            print('Edge weights undefined')
        return omega
    
    def getRho(self, paths, exp = 2):
        if self.__cFlag:
            numer = self.getEpsilon(paths, exp).sum()

            A = nx.incidence_matrix(self.G, oriented=True).toarray()
            m,n = A.shape
            b = np.zeros(m-1)
            b[0] = -1

            _,inds = sp.Matrix(A[:m-1]).rref()

            mask = np.ones(n,dtype=bool)
            mask[list(inds)] = False
            B = A[:m-1,~mask]
            N = A[:m-1,mask]
            H = np.append(-(np.linalg.inv(B)@N),np.identity(n-(m-1)),axis=0)
            b_H = np.append(-np.linalg.inv(B)@b,np.zeros(n-(m-1)))
            z = np.array(vectorizePaths(Prob.step_index,reference_paths,Prob.edges))[:,mask]
            denom = (((H@z.T).T-b_H)**2).sum()
            rho = max(1-numer/denom,0)
            return rho
        
        else: 
            print('Edge weights undefined')                        
    