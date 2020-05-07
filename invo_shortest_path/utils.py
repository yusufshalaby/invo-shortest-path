import cvxpy as cvx
import numpy as np
import networkx as nx
from collections import Counter

def badEdges(points, n):
    sumpoints = sum(points)
    return [j for j in range(n) if sumpoints[j]==0]

def infNorm(c,y,eps,obj,cons,n,solver='GUROBI'):
    results = {}
    for j in range(n):
        normcons = []
        normcons.append(c<=1)
        normcons.append(c>=-1)
        normcons.append(c[j]==1)
        prob1 = cvx.Problem(obj,cons + normcons)
        try:
            prob1.solve(solver=solver, verbose=False)
            results[1,j,'c'] = c.value
            results[1,j,'y'] = y.value
            results[1,j,'eps'] = eps.value
            results[1,j,'obj'] = obj.value
            #print(j,1,obj.value)
        except cvx.error.SolverError:
            results[1,j,'obj'] = 1e10

    for j in range(n):
        normcons = []
        normcons.append(c<=1)
        normcons.append(c>=-1)
        normcons.append(c[j]==-1)
        prob = cvx.Problem(obj,cons + normcons)
        try:
            prob.solve(solver=solver, verbose=False)
            results[-1,j,'c'] = c.value
            results[-1,j,'y'] = y.value
            results[-1,j,'eps'] = eps.value
            results[-1,j,'obj'] = obj.value
            #print(j,-1,obj.value)
        except cvx.error.SolverError: 
            results[-1,j,'obj'] = 1e10

    minobj = 1e10
    index1 = 0
    index2 = 0
    for i in [-1,1]:
        for j in range(n):
            if results[i,j,'obj'] < minobj:
                minobj = results[i,j,'obj']
                index1 = i
                index2 = j
    
    return results, index1, index2

def AbsoluteDualityGapPhase2(A,b,refpoints,goodpoints=[],badpoints=[],constraints=[],capconstraints=[],solver='GUROBI',err_tol=0,verbose=True):

    m,n = A.shape
    nGoodPoints = len(goodpoints)
    nBadPoints = len(badpoints)
    y = cvx.Variable(m)
    c = cvx.Variable(n)
   
    basecons = []
    basecons.append(A.T * y <= c)
    basecons.append(y[m-1] == 0)
    basecons.append(A*c == 0)
    
    for constraint in constraints:
        basecons.append(cvx.sum_entries(c[constraint[0]]) <= cvx.sum_entries(c[constraint[1]]))
    for capconstraint in capconstraints:
        basecons.append(c[capconstraint[0]] >= capconstraint[1])
    if nBadPoints==0:
        for j in badEdges(refpoints,n):
            basecons.append(c[j] >= 0)
        #for j in goodEdges(points,n):
        #    basecons.append(c[j] <= 0)
        
    normcons = []
    normcons.append(c<=1)
    normcons.append(c>=-1)
    
    if (nGoodPoints>0) | (nBadPoints>0):
        phase2cons = []
        phase2cons.append(c.T*refpoints - b.T*y <= 0+err_tol)
        
        if (nGoodPoints>0) & (nBadPoints>0):
            eps_good = cvx.Variable(nGoodPoints)
            eps_bad = cvx.Variable(nBadPoints)
            obj2 = cvx.Minimize((nBadPoints/nGoodPoints)*cvx.sum_entries(eps_good)-cvx.sum_entries(eps_bad))
            phase2cons.append(eps_good.T == c.T*goodpoints - b.T*y)
            phase2cons.append(eps_bad.T == c.T*badpoints - b.T*y)
            
        elif (nGoodPoints>0):
            eps_good = cvx.Variable(nGoodPoints)
            obj2 = cvx.Minimize(cvx.sum_entries(eps_good))
            phase2cons.append(eps_good.T == c.T*goodpoints - b.T*y)
            
        elif (nBadPoints>0):
            eps_bad = cvx.Variable(nBadPoints)
            obj2 = cvx.Maximize(cvx.sum_entries(eps_bad))
            phase2cons.append(eps_bad.T == c.T*badpoints - b.T*y)
        
        phase2 = cvx.Problem(obj2,basecons + phase2cons + normcons)
        phase2.solve(solver=solver, verbose=verbose)
        
    clist = [np.asarray(cval)[0].tolist()[0] for cval in c.value]
    ylist = [np.asarray(yval)[0].tolist()[0] for yval in y.value]
            
    return clist, ylist
        

def AbsoluteDualityGapPhases(A,b, refpoints, goodpoints=[], badpoints=[], constraints=[],capconstraints=[], exp=2, solver='GUROBI', normIndex=None, err_tol=0,verbose=True):    
    m,n = A.shape
    nRefPoints = len(refpoints)
    nGoodPoints = len(goodpoints)
    nBadPoints = len(badpoints)
    y = cvx.Variable(m)
    c = cvx.Variable(n)
    eps_ref_1 = cvx.Variable(nRefPoints)

    obj1 = cvx.Minimize(cvx.sum_entries(eps_ref_1**exp))
    
    #obj = cvx.Minimize(cvx.sum_entries(z0**exp))
    basecons = []
    basecons.append(A.T * y <= c)
    basecons.append(y[m-1] == 0)
    basecons.append(A*c == 0)
    phase1cons = []
    phase1cons.append(eps_ref_1.T == c.T*refpoints - b.T*y)
    
    for constraint in constraints:
        basecons.append(cvx.sum_entries(c[constraint[0]]) <= cvx.sum_entries(c[constraint[1]]))
    for capconstraint in capconstraints:
        basecons.append(c[capconstraint[0]] >= capconstraint[1])
    if nBadPoints==0:
        for j in badEdges(refpoints,n):
            basecons.append(c[j] >= 0)
        #for j in goodEdges(points,n):
        #    basecons.append(c[j] <= 0)
        
    if isinstance(normIndex,int):
        normcons = []
        normcons.append(c<=1)
        normcons.append(c>=-1)
        normcons.append(c[normIndex]==-1)

        phase1 = cvx.Problem(obj1,basecons + phase1cons + normcons)
        #prob.solve(solver=solver,abstol=1e-15, feastol=1e-15, max_iters=10000, verbose=True)
        phase1.solve(solver=solver, verbose=verbose)
        
        if (nGoodPoints>0) | (nBadPoints>0):
            eps_ref_2 = cvx.Variable(nRefPoints)
            phase2cons = []
            phase2cons.append(eps_ref_2 <= eps_ref_1.value+err_tol)
            phase2cons.append(eps_ref_2.T == c.T*refpoints - b.T*y)
            
            if (nGoodPoints>0) & (nBadPoints>0):
                eps_good = cvx.Variable(nGoodPoints)
                eps_bad = cvx.Variable(nBadPoints)
                obj2 = cvx.Minimize((nBadPoints/nGoodPoints)*cvx.sum_entries(eps_good)-cvx.sum_entries(eps_bad))
                phase2cons.append(eps_good.T == c.T*goodpoints - b.T*y)
                phase2cons.append(eps_bad.T == c.T*badpoints - b.T*y)
                
            elif (nGoodPoints>0):
                eps_good = cvx.Variable(nGoodPoints)
                obj2 = cvx.Minimize(cvx.sum_entries(eps_good))
                phase2cons.append(eps_good.T == c.T*goodpoints - b.T*y)
                
            elif (nBadPoints>0):
                eps_bad = cvx.Variable(nBadPoints)
                obj2 = cvx.Maximize(cvx.sum_entries(eps_bad))
                phase2cons.append(eps_bad.T == c.T*badpoints - b.T*y)
            
            phase2 = cvx.Problem(obj2,basecons + phase2cons + normcons)
            phase2.solve(solver=solver, verbose=verbose)
            
        clist = [np.asarray(cval)[0].tolist()[0] for cval in c.value]
        ylist = [np.asarray(yval)[0].tolist()[0] for yval in y.value]
            
        return clist, ylist
    
    results,index1,index2 = infNorm(c,y,eps_ref_1,obj1,basecons+phase1cons,n)
    if (index1 == 0): print('Phase 1 unsolved')
    print(results[index1,index2,'eps'])
    print(results[index1,index2,'obj'])
     
    if (nGoodPoints>0) | (nBadPoints>0):
            eps_ref_2 = cvx.Variable(nRefPoints)
            phase2cons = []
            phase2cons.append(eps_ref_2 <= eps_ref_1.value+err_tol)
            phase2cons.append(eps_ref_2.T == c.T*refpoints - b.T*y)
            
            if (nGoodPoints>0) & (nBadPoints>0):
                eps_good = cvx.Variable(nGoodPoints)
                eps_bad = cvx.Variable(nBadPoints)
                obj2 = cvx.Minimize(((nBadPoints/nGoodPoints)*cvx.sum_entries(eps_good)-cvx.sum_entries(eps_bad)))
                phase2cons.append(eps_good.T == c.T*goodpoints - b.T*y)
                phase2cons.append(eps_bad.T == c.T*badpoints - b.T*y)
                
            elif (nGoodPoints>0):
                eps_good = cvx.Variable(nGoodPoints)
                obj2 = cvx.Minimize(cvx.sum_entries(eps_good**exp))
                phase2cons.append(eps_good.T == c.T*goodpoints - b.T*y)
                
            elif (nBadPoints>0):
                eps_bad = cvx.Variable(nBadPoints)
                obj2 = cvx.Maximize(cvx.sum_entries(eps_bad**exp))
                phase2cons.append(eps_bad.T == c.T*badpoints - b.T*y)
            
            results,index1,index2 = infNorm(c,y,eps_ref_2,obj2,basecons+phase2cons,n)
            if (index1 == 0): print('Phase 2 unsolved. Increase err_tol.')
            print(results[index1,index2,'eps'])
            print(results[index1,index2,'obj'])
            print(index2)
    
    clist = results[index1,index2,'c']
    clist = [np.asarray(cval)[0].tolist()[0] for cval in clist]
    ylist = results[index1,index2,'y']
    ylist = [np.asarray(yval)[0].tolist()[0] for yval in ylist]
    
    return clist, ylist


def AbsoluteDualityGap(A,b, refpoints, constraints=[],capconstraints=[], exp=2, solver='GUROBI', normIndex=None):  
    results = {}
    m,n = A.shape
    nRefPoints = len(refpoints)
    y = cvx.Variable(m)
    c = cvx.Variable(n)
    z0 = cvx.Variable(nRefPoints)
    obj = cvx.Minimize(cvx.sum_entries(z0**exp))
    
    #obj = cvx.Minimize(cvx.sum_entries(z0**exp))
    basecons = []
    basecons.append(A.T * y <= c)
    basecons.append(y[m-1] == 0)
    basecons.append(A*c == 0)
    basecons.append(z0.T == c.T*refpoints - b.T*y)

    
    for constraint in constraints:
        basecons.append(cvx.sum_entries(c[constraint[0]]) <= cvx.sum_entries(c[constraint[1]]))
    for capconstraint in capconstraints:
        basecons.append(c[capconstraint[0]] >= capconstraint[1])
    for j in badEdges(refpoints,n):
            basecons.append(c[j] >= 0)
        #for j in goodEdges(points,n):
        #    basecons.append(c[j] <= 0)
        
    if isinstance(normIndex,int):
        basecons.append(c<=1)
        basecons.append(c>=-1)
        basecons.append(c[normIndex]==-1)
        prob = cvx.Problem(obj,basecons)
        #prob.solve(solver=solver,abstol=1e-15, feastol=1e-15, max_iters=10000, verbose=True)
        prob.solve(solver=solver, verbose=True)
        clist = [np.asarray(cval)[0].tolist()[0] for cval in c.value]
        ylist = [np.asarray(yval)[0].tolist()[0] for yval in y.value]
        return clist, ylist
    
    for j in range(n):
        normcons = []
        normcons.append(c<=1)
        normcons.append(c>=-1)
        normcons.append(c[j]==1)
        cons = basecons+normcons
        prob = cvx.Problem(obj,cons)
        try:
            prob.solve(solver=solver, verbose=False)
            results[1,j,'c'] = c.value
            results[1,j,'y'] = y.value
            results[1,j,'obj'] = obj.value
            #print(j,1,obj.value)
        except cvx.error.SolverError:
            print("solver goofed")
            results[1,j,'obj'] = 1e10

    for j in range(n):
        normcons = []
        normcons.append(c<=1)
        normcons.append(c>=-1)
        normcons.append(c[j]==-1)
        cons = basecons+normcons
        prob = cvx.Problem(obj,cons)
        try:
            prob.solve(solver=solver, verbose=False)
            results[-1,j,'c'] = c.value
            results[-1,j,'y'] = y.value
            results[-1,j,'obj'] = obj.value
            #print(j,-1,obj.value)
        except cvx.error.SolverError:
            print("solver goofed") 
            results[-1,j,'obj'] = 1e10

    minobj = 1e10
    index1 = 0
    index2 = 0
    for i in [-1,1]:
        for j in range(n):
            if results[i,j,'obj'] < minobj:
                minobj = results[i,j,'obj']
                index1 = i
                index2 = j
            
    clist = results[index1,index2,'c']
    clist = [np.asarray(cval)[0].tolist()[0] for cval in clist]
    ylist = results[index1,index2,'y']
    ylist = [np.asarray(yval)[0].tolist()[0] for yval in ylist]
    
    return clist, ylist

def optimizeFOPdual(G,c):
    A = nx.incidence_matrix(G, oriented=True).toarray()
    m,n = A.shape
    y = cvx.Variable(m)
    cons = []
    cons.append(A.T*y <= c)
    cons.append(y[m-1]==0)
    obj = (cvx.Maximize(y[m-1]-y[0]))
    prob = cvx.Problem(obj, cons)
    prob.solve(solver='GUROBI',verbose=False,abstol=0)
    objval = prob.value
    dual = y.value
    dual = np.array([dual[i,0] for i in range(len(dual))])
    opt = cons[0].dual_value
    opt = np.array([opt[i,0] for i in range(len(opt))])
    optpath = list(np.flatnonzero(opt >.001))
    return objval, opt, optpath, dual

def levenshtein(seq1, seq2):  
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros ((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    return matrix, (matrix[size_x - 1, size_y - 1])

def vectorizePaths(singlestatenetwork, paths):
    n = len(singlestatenetwork.edges)
    points = []
    for path in paths:
        point = np.zeros(n)
        outer_edges = [(singlestatenetwork.step_index[node_1][-1],singlestatenetwork.step_index[node_2][0]) for node_1,node_2 in zip(path[:-1],list(path[1:]))]
        inner_edges = [(singlestatenetwork.step_index[step][0],singlestatenetwork.step_index[step][1]) for step in path[1:] if len(singlestatenetwork.step_index[step])==2]
        end_edge = [(singlestatenetwork.step_index[path[-1]][-1],singlestatenetwork.step_index['END'][0])] if path[-1] in singlestatenetwork.end_steps else []
        
        edge_indices = [singlestatenetwork.edges.index(edge) for edge in outer_edges+inner_edges+end_edge]
        edge_indices_dict = Counter(edge_indices)
        point[list(edge_indices_dict.keys())] = list(edge_indices_dict.values())

        points.append(point)
    return points

def vectorizeStatePaths(multistatenetwork,statepaths):
    n = len(multistatenetwork.edges)
    points = []
    end_order = max([len(state) for state in multistatenetwork.states])+1
    for statepath in statepaths:
        point = np.zeros(n)
        state_order = [end_order if state=='END' else len(state) for step,state in statepath]
        step_order = [0 if step=='START' else i for i,(step,state) in enumerate(statepath)]
        intermediate_steps = [(statepath[i-1][0]+' START',statepath[i][1]) for i in range(1,len(statepath)) if (statepath[i-1][1]!=statepath[i][1]) & (statepath[i][1]!='END')]
        intermediate_state_order = [len(state) for step,state in intermediate_steps]
        intermediate_step_order = [0]*len(intermediate_steps)

        statepath_full = list(zip(statepath,state_order,step_order)) + list(zip(intermediate_steps,intermediate_state_order,intermediate_step_order))
        statepath_full = sorted(statepath_full,key=lambda tup:(tup[1],tup[2]))
        statepath_full = [item[0] for item in statepath_full]

        outer_edges = [(multistatenetwork.state_name_dict[node_1[1]]+str(multistatenetwork.state_network_dict[node_1[1]].step_index[node_1[0]][-1]), 
            multistatenetwork.state_name_dict[node_2[1]]+str(multistatenetwork.state_network_dict[node_2[1]].step_index[node_2[0]][0])) 
                                                    for node_1,node_2 in zip(statepath_full[:-1],statepath_full[1:])]
        inner_edges = [(multistatenetwork.state_name_dict[stepstate[1]]+str(multistatenetwork.state_network_dict[stepstate[1]].step_index[stepstate[0]][0]),
            multistatenetwork.state_name_dict[stepstate[1]]+str(multistatenetwork.state_network_dict[stepstate[1]].step_index[stepstate[0]][1])) 
                                                    for stepstate in statepath_full[1:] if len(multistatenetwork.state_network_dict[stepstate[1]].step_index[stepstate[0]])==2]

        destin_step = statepath[-1][0]
        destin_state = statepath[-1][1]
        if destin_state == 'END':
            end_edge = []
        else:
            resultant_state = 'END' if destin_step in multistatenetwork.end_steps else tuple(sorted(destin_state + (destin_step,)))
            if resultant_state in multistatenetwork.states:       
                resultant_step = 'END' if resultant_state == 'END' else destin_step+' START'
                final_destination = multistatenetwork.state_name_dict[resultant_state] + str(multistatenetwork.state_network_dict[resultant_state].step_index[resultant_step][0])
                end_edge = [(outer_edges[-1][1],final_destination)]
            else:
                end_edge = []

        edge_indices = [multistatenetwork.edges.index(edge) for edge in outer_edges+inner_edges+end_edge]
        edge_indices_dict = Counter(edge_indices)
        point[list(edge_indices_dict.keys())] = list(edge_indices_dict.values())

        points.append(point)
    return points  

