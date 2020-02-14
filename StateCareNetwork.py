# -*- coding: utf-8 -*-
"""
Created on Wed Jan 22 15:19:41 2020

@author: yusuf
"""
import networkx as nx
from itertools import combinations
from CareNetwork import CareNetwork
from utils import vectorizeStatePaths, AbsoluteDualityGapPhases, optimizeFOPdual
import numpy as np

class StateCareNetwork():
    
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
        unique_path_states = [set(path_states[i]).difference(set([step for j in range(len(path_states)) 
                                        for step in path_states[j] if j != i])) for i in range(len(path_states))]
        common_path_states = set.intersection(*[set(path_state) for path_state in path_states]).union({()})
        
        for i in range(len(reference_paths)):
            for state in unique_path_states[i]:
                self.state_steps_dict[state] = []
                discordant_steps = rep_steps.difference(path_steps[i])
                state_steps = set(state).intersection(rep_steps).union(discordant_steps)
                final_end_steps = end_steps if any(trigger_step in state for trigger_step in trigger_steps) else {}
                state_end_steps = path_steps[i].difference(set(state)).union(final_end_steps)
                self.state_steps_dict[state].append(sorted(list(state_steps)))
                self.state_steps_dict[state].append(sorted(list(state_end_steps)))
        
        for state in common_path_states:
            self.state_steps_dict[state] = []
            discordant_steps = rep_steps.difference(path_steps[0].union(path_steps[1]))
            state_steps = set(state).intersection(rep_steps).union(discordant_steps)
            final_end_steps = end_steps if any(trigger_step in state for trigger_step in trigger_steps) else {}
            state_end_steps = path_steps[0].union(path_steps[1]).difference(set(state)).union(final_end_steps)
            self.state_steps_dict[state].append(sorted(list(state_steps)))
            self.state_steps_dict[state].append(sorted(list(state_end_steps)))
            
        self.states = set(self.state_steps_dict.keys())
        
        self.state_network_dict = {}
        for state in self.states:
            self.state_network_dict[state] = CareNetwork()
            state_steps = self.state_steps_dict[state][0]
            state_start_steps = [start_step + ' START' for start_step in state]
            state_end_steps = self.state_steps_dict[state][1]
            start_dummy = state == ()
            self.state_network_dict[state].formNetwork(steps=state_steps,start_steps=state_start_steps,end_steps=state_end_steps, start_step_nodes=1, end_step_nodes=1, start_end_connect =True,
                                             start_dummy = start_dummy, end_dummy = False)
        self.state_network_dict[('END')] = CareNetwork()
        self.state_network_dict[('END')].formNetwork(steps=[],start_steps=[],end_steps=[], start_dummy = False, end_dummy = True)
        
        self.refstatepaths = []
        for path in reference_paths:    
            refstatepath = [('START',()),(path[1],())]
            completed = []
            for i in range(2,len(path)):
                completed.append(path[i-1])
                state = tuple(sorted(completed))
                refstatepath.append((path[i],state))
            refstatepath.append(('END','END'))
            self.refstatepaths.append(refstatepath)
        
       
        
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
        self.__A = nx.incidence_matrix(self.G, oriented=True).toarray()
        self.__b = self.G.number_of_nodes()
        self.__b[list(self.G.nodes).index('s 0')] = -1
        self.__b[list(self.G.nodes).index('t 0')] = 1
        
        
    def showPath(self, statepath):
        state_progression = [state for state in list(dict.fromkeys([state for step,state in statepath]))]
        
        sol_edges = np.array(self.edges)[np.flatnonzero(vectorizeStatePaths(self.state_network_dict, self.state_name_dict, [statepath], self.edges))]
        statepath_grouped = list(zip(state_progression,[np.array([list(self.state_network_dict[current_state].G.edges).index((self.node_state_dict[node1][1],
        self.node_state_dict[node2][1])) for (node1,node2) in sol_edges if (self.node_state_dict[node1][0] == current_state) & 
        (self.node_state_dict[node2][0] == current_state)]) for current_state in state_progression]))
        
        fig_dict = {}
        for state,path in statepath_grouped:
            fig_dict[state] = self.state_network_dict[state].showNetwork(path = path)
            
    def getCoefficients(self,goodstatepaths,badstatepaths,err_tol=0):
        refpoints = vectorizeStatePaths(self.state_network_dict, self.state_name_dict, self.refstatepaths, self.edges) 
        goodpoints = vectorizeStatePaths(self.state_network_dict, self.state_name_dict, goodstatepaths, self.edges) 
        badpoints = vectorizeStatePaths(self.state_network_dict, self.state_name_dict, badstatepaths, self.edges) 


        c, y = AbsoluteDualityGapPhase2(A=self.__A, b=self.__b, refpoints=refpoints, goodpoints = goodpoints, 
                                        badpoints = badpoints,err_tol=err_tol)

        c = np.asarray(c)
        y = np.asarray(y)

        self.setEdgeWeights(c, y)
        
    def setEdgeWeights(self, c, y = []):
        if (len(c) == self.G.number_of_edges()):
            for e in self.edges:
                self.G[e[0]][e[1]]['weight'] = c[self.edges.index(e)]
            self.c = c
            if len(y) == self.G.number_of_nodes():
                self.y = y
            else:
                _,_,_, self.y = optimizeFOPdual(self.G,c)
        else: print('Wrong cost vector dimensions')
        
    
        
        
        
        
        