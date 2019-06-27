import cvxpy as cvx
import numpy as np

def AbsoluteDualityGap(A,b, refpoints, goodpoints=[], badpoints=[], constraints=[],capconstraints=[], exp=2, solver='GUROBI', normIndex=None, reg_par=0):    
    results = {}
    m,n = A.shape
    nRefPoints = len(refpoints)
    nGoodPoints = len(goodpoints)
    nBadPoints = len(badpoints)
    y = cvx.Variable(m)
    z0 = cvx.Variable(nRefPoints)
    z1 = cvx.Variable(nGoodPoints)
    z2 = cvx.Variable(nBadPoints)
    c = cvx.Variable(n)
    obj = cvx.Minimize(cvx.sum_entries(z0**exp)+reg_par*((nBadPoints/nGoodPoints)*cvx.sum_entries(z1)-cvx.sum_entries(z2)))
    #obj = cvx.Minimize(cvx.sum_entries(z0**exp))
    basecons = []
    basecons.append(A.T * y <= c)
    basecons.append(y[m-1] == 0)
    basecons.append(A*c == 0)
    for q in range(nRefPoints):
        basecons.append(z0[q] == c.T*refpoints[q] - b.T*y)
    for q in range(nGoodPoints):
        basecons.append(z1[q] == c.T*goodpoints[q] - b.T*y)
    for q in range(nBadPoints):
        basecons.append(z2[q] == c.T*badpoints[q] - b.T*y)

    
    for constraint in constraints:
        basecons.append(cvx.sum_entries(c[constraint[0]]) <= cvx.sum_entries(c[constraint[1]]))
    for capconstraint in capconstraints:
        basecons.append(c[capconstraint[0]] >= capconstraint[1])
    if nBadPoints==0:
        for j in badEdges(points,n):
            basecons.append(c[j] >= 0)
        #for j in goodEdges(points,n):
        #    basecons.append(c[j] <= 0)
        
    if isinstance(normIndex,int):
        basecons.append(c<=1)
        basecons.append(c>=-1)
        basecons.append(c[normIndex]==-1)
        prob = cvx.Problem(obj,basecons)
        #prob.solve(solver=solver,abstol=1e-15, feastol=1e-15, max_iters=10000, verbose=True)
        prob.solve(solver=solver, BarQCPConvTol = 1e-10, verbose=False)
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
            prob.solve(solver=solver, BarQCPConvTol=1e-10, verbose=False)
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
            prob.solve(solver=solver, BarQCPConvTol=1e-10, verbose=False)
            results[-1,j,'c'] = c.value
            results[-1,j,'y'] = y.value
            results[-1,j,'obj'] = obj.value
            #print(j,-1,obj.value)
        except cvx.error.SolverError:
            print("solver goofed") 
            results[-1,j,'obj'] = 1e10

    objs = []
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

def vectorizePaths(step_index, paths, edges):
    N=len(step_index)-2
    n = len(edges)
    points = []
    for path in paths:
        point = np.zeros(n)
        for i in range(1,len(path)-1):
            origin = step_index[path[i-1]][-1]
            destination = step_index[path[i]][0]
            edge = (origin, destination)
            point[edges.index(edge)] += 1
            if len(step_index[path[i]])>1:
                edge = (destination, step_index[path[i]][-1])
                point[edges.index(edge)] += 1
        origin = step_index[path[-2]][-1]
        destination = step_index[path[-1]][0]
        edge = (origin,destination)
        point[edges.index(edge)] += 1
        points.append(point)
    return points

