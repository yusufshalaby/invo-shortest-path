import networkx as nx
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import pandas as pd
from itertools import combinations

from InverseModels import AbsoluteDualityGap, AbsoluteDualityGapPhase2, optimizeFOPdual, vectorizePaths, vectorizeStatePaths

class InverseShortestPathModel():
            
    def setEdgeWeights(self, c, y = []):
        if self._networkFlag:
            if (len(c) == self.G.number_of_edges()):
                for e in self.edges:
                    self.G[e[0]][e[1]]['weight'] = c[self.edges.index(e)]
                self.c = c
                if len(y) == self.G.number_of_nodes():
                    self.y = y
                else:
                    _,_,_, self.y = optimizeFOPdual(self.G,c)
                self._cFlag = True
            else: print('Wrong cost vector dimensions')
        else: self._missingNetworkError()
        
    def setEdgeHist(self, h):
        if self._networkFlag:
            if (len(h) == self.G.number_of_edges()):
                for e in self.edges:
                    self.G[e[0]][e[1]]['hist'] = h[self.edges.index(e)]
                self.h = h
                self._hFlag = True
            else: print('Wrong vector dimensions')
        else: self._missingNetworkError()
        
    def solveIO(self, refpoints, goodpoints = [], badpoints = [], constraints = [], 
                        capconstraints = [], solver = 'GUROBI',err_tol=0, verbose = True, normIndex=None, exp=1):
      
        if (self.startDummy) & (self.endDummy):

            if (len(goodpoints)>0) | (len(badpoints)>0):
                c, y = AbsoluteDualityGapPhase2(A=self._A, b=self._b, refpoints=refpoints, goodpoints=goodpoints,
                                             badpoints=badpoints, constraints=constraints, capconstraints=capconstraints,
                                             solver = solver, err_tol=err_tol,verbose=verbose)
            else:
                c, y = AbsoluteDualityGap(A=self._A,b=self._b, refpoints=refpoints, constraints=constraints,capconstraints=capconstraints, exp=exp, solver='GUROBI', normIndex=normIndex)

            c = np.asarray(c)
            y = np.asarray(y)

            self.setEdgeWeights(c, y)
        
        else: self._missingNetworkError()
        
    def getEpsilon(self, points, exp=1):
        
        if self._cFlag:
            obj = self.y[np.flatnonzero(self._b==1)[0]]-self.y[np.flatnonzero(self._b==-1)[0]]
            epsilon = (np.dot(points, self.c) - obj)**exp
            return epsilon
        else: 
            print('Edge weights undefined')
    
    def maxDistances(self,edgelength):
           
        D = {}
        P = {}
        nodelength = edgelength+1
        start = self.nodes[np.flatnonzero(self._b==-1)[0]]
        end = self.nodes[np.flatnonzero(self._b==1)[0]]
        D[start,0]=0
        #for v in nodes: 
        #    if v != start: 
        #        D[v,1]=M
        
       
        for k in range(1,nodelength+1):
            for v in self.nodes:
                candidates = {}
                for u in self.G.predecessors(v):
                    #if (u,v) in self.edges:
                    if (u,k-1) in D:
                        candidates[u] = D[u,k-1]-self.G[u][v]['weight']
                    #else:
                    #    candidates[u] = D[u,k-1]+M
                    if candidates:
                        u_min = min(candidates, key=candidates.get)
                        D[v,k] = candidates[u_min]  
                        P[v,k] = u_min
        
        Dmax = {k:-D[end,k] for (node,k) in D.keys() if node == end}

        return Dmax

    def getOmega(self, points, exp = 1):
        
        if self._cFlag: 
            obj = self.y[np.flatnonzero(self._b==1)[0]]-self.y[np.flatnonzero(self._b==-1)[0]]

            pathlengths = np.sum(points,axis=1)
            maxlength = np.int(max(pathlengths))
            Dmax = self.maxDistances(maxlength)

            numerator = self.getEpsilon(points,exp)
            denom = (np.array([Dmax[length]-obj for length in pathlengths]))**exp
            omega = 1-numerator/denom
        else:
            print('Edge weights undefined')
        return omega
    
    def getRho(self, points, exp = 2):
        if self._cFlag:
            numer = self.getEpsilon(points, exp).sum()

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
            z = np.array(points)[:,mask]
            denom = (((H@z.T).T-b_H)**2).sum()
            rho = max(1-numer/denom,0)
            return rho
        
        else: 
            print('Edge weights undefined')      
        

class SingleStateModel(InverseShortestPathModel):
    
    def __init__(self, steps, start_steps=[], end_steps=[], start_step_nodes=2, end_step_nodes=2, start_end_connect=False,
                    start_dummy=True, end_dummy=True):
        self._cFlag = False
        self._hFlag = False
        self.formNetwork(steps, start_steps, end_steps, start_step_nodes, end_step_nodes, start_end_connect,
                    start_dummy, end_dummy)

    
    def formNetwork(self, steps, start_steps=[], end_steps=[], start_step_nodes=2, end_step_nodes=2, start_end_connect=False,
                    start_dummy=True, end_dummy=True):
        self.G = nx.DiGraph()
        self.pos = {}
        N0 = len(start_steps)
        N1 = len(steps)
        N2 = len(end_steps)
        
        G0 = nx.complete_graph(N1)
        pos1 = nx.circular_layout(G0)
        pos2 = {}
        for p in pos1:
            pos2[p]=[pos1[p][0]-0.2, pos1[p][1]]

        flipped_pos1 = {node: (-x,y) for (node, (x,y)) in pos1.items()}
        flipped_pos2 = {node: (-x,y) for (node, (x,y)) in pos2.items()}
        pos1 = list(flipped_pos1.values())
        pos2 = list(flipped_pos2.values())

        poslist= ([(-1.9,0)] + [p for i in range(start_step_nodes) for p in list(zip((-1.6+0.2*i)*np.ones(N0),np.linspace(-1,1,N0)))] +
                                pos1 + pos2 +
                                [p for i in range(end_step_nodes) for p in list(zip((1.7+0.2*i)*np.ones(N2),np.linspace(-1,1,N2)))] +
                                [(2.0,0)]) 
        
        
        pos_index_end = len(poslist) 
        
        
        self.step_index = {}
        if start_dummy:
            self.step_index['START'] = [0]
            a = 1
            pos_index_start = 0
        else: 
            a = 0
            pos_index_start = 1
        
        self.step_index = dict(self.step_index, ** {step:[a+i*N0+start_steps.index(step) 
                                                    for i in range(start_step_nodes)] for step in start_steps},
                                                ** {step:[a+start_step_nodes*N0+i*N1+steps.index(step)
                                                    for i in range(2)] for step in steps},
                                                ** {step:[a+start_step_nodes*N0+2*N1+i*N2+end_steps.index(step)
                                                    for i in range(end_step_nodes)] for step in end_steps})
                
        if end_dummy:
            b = 1
            self.step_index['END'] = [a+2*N0+2*N1+2*N2]
        else:
            b = 0
            pos_index_end -= 1
        
        poslist = poslist[pos_index_start:pos_index_end]

        self.labels = {self.step_index[step][0]:step for step in self.step_index.keys()}
        self.node_step_dict = {index:step for step in self.step_index.keys() for index in self.step_index[step]}



        self.G.add_nodes_from(range(start_step_nodes*N0+2*N1+end_step_nodes*N2+a+b))
        self.pos = dict((node,nodepos) for node, nodepos in zip(self.G.nodes(),poslist))
    
        for start_step in start_steps:
            if start_dummy: self.G.add_edge(self.step_index['START'][-1],self.step_index[start_step][0])
            for i in range(1,start_step_nodes):
                self.G.add_edge(self.step_index[start_step][i-1],self.step_index[start_step][i])
            for step in steps:
                self.G.add_edge(self.step_index[start_step][-1],self.step_index[step][0])
            if start_end_connect:
                for end_step in end_steps:
                    self.G.add_edge(self.step_index[start_step][-1],self.step_index[end_step][0])
        for step in steps:
            if (len(start_steps)==0) & (start_dummy):
                self.G.add_edge(self.step_index['START'][-1],self.step_index[step][0])
            self.G.add_edge(self.step_index[step][0],self.step_index[step][-1])
            for step2 in steps:
                self.G.add_edge(self.step_index[step][-1],self.step_index[step2][0])
            if (len(end_steps)==0) & (end_dummy):
                self.G.add_edge(self.step_index[step][-1],self.step_index['END'][0])
        for end_step in end_steps:
            if (start_dummy) & (start_end_connect): self.G.add_edge(self.step_index['START'][-1],self.step_index[end_step][0])
            for step in steps:
                self.G.add_edge(self.step_index[step][-1],self.step_index[end_step][0])
            for i in range(1,end_step_nodes):
                self.G.add_edge(self.step_index[end_step][i-1],self.step_index[end_step][i])
            if end_dummy: self.G.add_edge(self.step_index[end_step][-1],self.step_index['END'][0])
        self.edges = list(self.G.edges())
        self.nodes = list(self.G.nodes())
        
        self._N = len(self.step_index.keys())- (a+b)
        self._A = nx.incidence_matrix(self.G, oriented=True).toarray()
        self._b = np.zeros(self.G.number_of_nodes())
        if start_dummy: self._b[0] = -1
        if end_dummy: self._b[-1] = 1
        self._networkFlag = True
        self.startDummy = 1*start_dummy
        self.endDummy = 1*end_dummy
        self.end_steps = end_steps
                
    def showNetwork(self,show_labels=True,show_weights='no',path=None):
        if self._networkFlag:
            fig = plt.figure(figsize=(16,8))
            ax = fig.add_subplot(111, aspect='equal')
            
            n = self.G.number_of_nodes()
            visibleNodes = self.G.nodes()

            # draw the regular interior nodes in the graph
            nx.draw_networkx_nodes(self.G,self.pos,nodelist=visibleNodes,node_color='grey',node_size=100,ax=ax, alpha=0.2)

            # draw the origin and destination nodes
            dummy_nodes = list(zip([[0],[n-1]],['g','r']))
            for nodes, color in dummy_nodes[1-self.startDummy:1+self.endDummy]:
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

            if (show_weights == 'c') & (self._cFlag):
                p = [self.c[i] for i in visibleEdges] 
            elif (show_weights == 'h') & (self._hFlag):
                p = [self.h[i] for i in visibleEdges] 
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
                if isinstance(path,list): 
                    path = np.flatnonzero(vectorizePaths(self,[path])[0])
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

            if (show_weights == 'c') & (self._cFlag):
                cbar = fig.colorbar(edges)
                cbar.set_label(r'    $c_{ij}$',rotation=0,fontsize=18)
            elif (show_weights == 'h') & (self._hFlag):
                cbar = fig.colorbar(edges)
                cbar.set_label(r'    $h_{ij}$',rotation=0,fontsize=18)


            if show_labels:
                labelspos = []
                for i in range(len(self.pos)):
                    labelspos.append([self.pos[i][0]+0.1,self.pos[i][1]+0.05])
                nx.draw_networkx_labels(self.G,labelspos,self.labels,font_weight = 'bold', font_size=12)

            ax.axis([-2.2,2.2,-1.2,1.2])
            #ax.axis('tight')
            #ax.axis('equal')
            ax.axis('off')
            
            return fig
            
            

        else: self._missingNetworkError()
    
    def getConstraints(self, step_ranks = [], subpath_constraints = [], penalty_constraints = []):
            constraints = []
            capconstraints = []

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
                    indices = list(np.flatnonzero(vectorizePaths(self,[component])[0]>0))
                    constraint.append(indices)
                constraints.append(constraint)

            for penalty_constraint in penalty_constraints:
                constraint = []
                indices = list(np.flatnonzero(vectorizePaths(self,[penalty_constraint[0]])[0]>0))
                constraint.append(indices)
                constraint.append(penalty_constraint[1])
                capconstraints.append(constraint) 
                
            return constraints, capconstraints
        
    def getCoefficients(self,refpaths,goodpaths,badpaths,step_ranks=[],subpath_constraints=[],penalty_constraints=[],err_tol=0,verbose=True,normIndex=None,exp=1):
        constraints, capconstraints = self.getConstraints(step_ranks=step_ranks,subpath_constraints=subpath_constraints,penalty_constraints=penalty_constraints)
        refpoints = vectorizePaths(self, refpaths) 
        goodpoints = vectorizePaths(self, goodpaths) 
        badpoints = vectorizePaths(self, badpaths) 
        super().solveIO(refpoints=refpoints,goodpoints=goodpoints,badpoints=badpoints,constraints=constraints,capconstraints=capconstraints,verbose=verbose,normIndex=normIndex)
        
        
class MultiStateModel(InverseShortestPathModel):
    
    def __init__(self, rep_steps, nonrep_steps, reference_paths, end_steps, trigger_steps, step_abbr_dict):      
        self.formStateNetworks(rep_steps, nonrep_steps, reference_paths, end_steps, trigger_steps)
        self.formCompleteNetwork(end_steps, step_abbr_dict)
        
    def formStateNetworks(self, rep_steps, nonrep_steps, reference_paths, end_steps, trigger_steps):

        path_steps = []
        path_states = []
        
        for i in range(len(reference_paths)):
            states = []
            path_steps.append(set([step for step in reference_paths[i] if step in (rep_steps.union(nonrep_steps))]))
            for r in range(1,len(path_steps[i])+1):
                states += combinations(path_steps[i],r)
            path_states.append([tuple(sorted(state)) for state in states])

        self.state_steps_dict = {}
        self.unique_path_states = [set(path_states[i]).difference(set([step for j in range(len(path_states)) 
                                        for step in path_states[j] if j != i])) for i in range(len(path_states))]
        self.common_path_states = set.intersection(*[set(path_state) for path_state in path_states]).union({()})
        
        for i in range(len(reference_paths)):
            for state in self.unique_path_states[i]:
                self.state_steps_dict[state] = []
                discordant_steps = rep_steps.difference(path_steps[i])
                state_steps = set(state).intersection(rep_steps).union(discordant_steps)
                final_end_steps = end_steps if any(trigger_step in state for trigger_step in trigger_steps) else {}
                state_end_steps = path_steps[i].difference(set(state)).union(final_end_steps)
                self.state_steps_dict[state].append(sorted(list(state_steps)))
                self.state_steps_dict[state].append(sorted(list(state_end_steps)))
        
        for state in self.common_path_states:
            self.state_steps_dict[state] = []
            discordant_steps = rep_steps.difference(path_steps[0].union(path_steps[1]))
            state_steps = set(state).intersection(rep_steps).union(discordant_steps)
            final_end_steps = end_steps if any(trigger_step in state for trigger_step in trigger_steps) else {}
            state_end_steps = path_steps[0].union(path_steps[1]).difference(set(state)).union(final_end_steps)
            self.state_steps_dict[state].append(sorted(list(state_steps)))
            self.state_steps_dict[state].append(sorted(list(state_end_steps)))
                    
        self.state_network_dict = {}
        for state in self.state_steps_dict.keys():
            state_steps = self.state_steps_dict[state][0]
            state_start_steps = [start_step + ' START' for start_step in state]
            state_end_steps = self.state_steps_dict[state][1]
            start_dummy = state == ()
            self.state_network_dict[state] = SingleStateModel(steps=state_steps,start_steps=state_start_steps,end_steps=state_end_steps, start_step_nodes=1, end_step_nodes=1, start_end_connect =True,
                                             start_dummy = start_dummy, end_dummy = False)
        self.state_network_dict[('END')] = SingleStateModel(steps=[],start_steps=[],end_steps=[], start_dummy = False, end_dummy = True)
        self.states = set(self.state_network_dict.keys())
        
        self.refstatepaths = []
        for path in reference_paths:    
            refstatepath = [('START',()),(path[1],())]
            completed = []
            for i in range(2,len(path)-1):
                completed.append(path[i-1])
                state = tuple(sorted(completed))
                refstatepath.append((path[i],state))
            refstatepath.append(('END','END'))
            self.refstatepaths.append(refstatepath)
        
        self.end_steps = end_steps
       
        
    def formCompleteNetwork(self, end_steps, step_abbr_dict):
        
        self.state_name_dict = {state:(''.join(step_abbr_dict[step] for step in state if step!='')+' ') for state in self.state_network_dict.keys() if state!='END'}
        self.state_name_dict[()] = 's '
        self.state_name_dict['END'] = 't '
    
        self.G = nx.union_all([self.state_network_dict[state].G for state in self.states], 
                              rename = [self.state_name_dict[state] for state in self.states])
        
        for item in self.state_steps_dict.items():
            state = item[0]
            state_name = self.state_name_dict[state]
            for step in item[1][1]:
                node_1 = state_name + str(self.state_network_dict[state].step_index[step][-1])
                if step in end_steps: 
                    node_2 = 't 0'
                    self.G.add_edge(node_1,node_2)
                else:
                    new_state = tuple(sorted(set(state).union(set([step]))))
                    new_state_name = self.state_name_dict[new_state]
                    new_step = step + ' START'
                    node_2 = new_state_name + str(self.state_network_dict[new_state].step_index[new_step][0])
                    self.G.add_edge(node_1,node_2)
                    
        name_state_dict = {name[:-1]:state for state,name in self.state_name_dict.items()}
        self.node_state_dict = {node:(name_state_dict[name],int(node_index)) for (node,(name,node_index)) in [(node,str.split(node)) for node in self.G.nodes]}
        self.edges = list(self.G.edges)
        self.nodes = list(self.G.nodes)
        self._A = nx.incidence_matrix(self.G, oriented=True).toarray()
        self._b = np.zeros(self.G.number_of_nodes())
        self._b[list(self.G.nodes).index('s 0')] = -1
        self._b[list(self.G.nodes).index('t 0')] = 1
        self._networkFlag = True
        self.startDummy = True
        self.endDummy = True
        
        
    def showPath(self, statepath):
        state_progression = [state for state in list(dict.fromkeys([state for step,state in statepath]))]
        
        sol_edges = np.array(self.edges)[np.flatnonzero(vectorizeStatePaths(self, [statepath]))]
        statepath_grouped = list(zip(state_progression,[np.array([list(self.state_network_dict[current_state].G.edges).index((self.node_state_dict[node1][1],
        self.node_state_dict[node2][1])) for (node1,node2) in sol_edges if (self.node_state_dict[node1][0] == current_state) & 
        (self.node_state_dict[node2][0] == current_state)]) for current_state in state_progression]))
        
        fig_dict = {}
        for state,path in statepath_grouped:
            fig_dict[state] = self.state_network_dict[state].showNetwork(path = path)
        
        return fig_dict
            
    def getCoefficients(self,goodstatepaths,badstatepaths,err_tol=0,verbose=True):
        refpoints = vectorizeStatePaths(self, self.refstatepaths) 
        goodpoints = vectorizeStatePaths(self, goodstatepaths) 
        badpoints = vectorizeStatePaths(self, badstatepaths) 

        super().solveIO(refpoints = refpoints,goodpoints = goodpoints,badpoints = badpoints, verbose=verbose)
        

        
                   
    