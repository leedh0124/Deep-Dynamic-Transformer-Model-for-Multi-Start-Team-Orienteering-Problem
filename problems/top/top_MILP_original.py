# -*- coding: utf-8 -*-
"""
Created on Sat May 15 16:54:38 2021

@author: user
"""
# 1. The Team Orienteering Problem: Formulations and Branch-Cut and Price∗
# 2. file:///E:/Research/%EA%B0%9C%EC%9D%B8%EC%97%B0%EA%B5%AC/VRPP/mainchap_tvbis_2013.pdf -> Chapter 10: Vehicle Routing Problems with Profits
# 3. https://www.gurobi.com/documentation/9.1/examples/tsp_py.html

from gurobipy import *
import numpy as np

def solve_euclidian_dtop(depot, loc, prize, max_length, cur_loc, cur_tlen, max_vehicle_num, threads=0, timeout=None, gap=None):
    """
    Solves the Euclidan dTOP problem to optimality using the MIP formulation 
    with lazy subtour elimination constraint generation.
    :param points: list of (x, y) coordinate 
    :return: 
    """
    K = max_vehicle_num
    # points = [depot] + loc + each cur_loc
    # points.shape = (2,22,2)
    points = np.array([np.concatenate((depot[None,...], loc, cur_loc[n][None,...]), axis=0) for n in range(K)])
    _, n, _ = points.shape
    
    # Callback - use lazy constraints to eliminate sub-tours

    def subtourelim(model, where):
        if where == GRB.Callback.MIPSOL:
            # make a list of edges selected in the solution
            vals = model.cbGetSolution(model._vars)
            selected = tuplelist((i, j, k) for i, j, k in model._vars.keys() if vals[i, j, k] > 0.5)
            # find the shortest cycle in the selected edge list
            tour = subtour(selected)
            #if tour is not None:
            if not all([x == None for x in tour]):
                # add subtour elimination constraint for every pair of cities in tour
                # model.cbLazy(quicksum(model._vars[i, j]
                #                       for i, j in itertools.combinations(tour, 2))
                #              <= len(tour) - 1)
                assert len(tour) == K
                for k in range(K): # always two subtours? TODO CHECK
                    if tour[k] is not None:
                        model.cbLazy(quicksum(model._vars[i, j, k]
                                              for i, j in itertools.combinations(tour[k], 2))
                                     <= quicksum(model._dvars[i,k] for i in tour[k]) * (len(tour[k]) - 1) / float(len(tour[k])))

                
    # Given a tuplelist of edges, find the shortest subtour
    
    def subtour(edges_total, exclude_depot=True):
        all_cycle = []
        for k in range(K):
            edges = edges_total.select('*','*',k)
            unvisited = list(range(n))
            #cycle = range(n + 1)  # initial length has 1 more city
            cycle = None
            while unvisited:  # true if list is non-empty
                thiscycle = []
                neighbors = unvisited
                while neighbors:
                    current = neighbors[0]
                    thiscycle.append(current)
                    unvisited.remove(current)
                    neighbors = [j for i, j, k in edges.select(current, '*', k) if j in unvisited]
                # If we do not yet have a cycle or this is the shorter cycle, keep this cycle
                # Unless it contains the depot while we do not want the depot
                if (
                    (cycle is None or len(cycle) > len(thiscycle))
                        and len(thiscycle) > 1 and not (0 in thiscycle and exclude_depot)
                ):
                    cycle = thiscycle
            all_cycle.append(cycle)
        return all_cycle
    
    # Dictionary of Euclidean distance between each pair of points, excluding i=0 (n*(n+1)/2 = 231, n=20+1 per each cur_loc)
    dist = {(i,j,k) :
        math.sqrt(sum((points[k][i][w]-points[k][j][w])**2 for w in range(2)))
        for i in range(n) for j in range(i) for k in range(K)}
    
    # Create Model
    m = Model()
    m.Params.outputFlag = True

    # Create variables
    vars = m.addVars(dist.keys(), vtype=GRB.BINARY, name='x')
    for i,j,k in vars.keys():
        vars[j,i,k] = vars[i,j,k] # edge in opposite direction for each vehicle

    # Depot vars can be 2. 0818: Not necessary since now we start at cur_loc, not depot
    #for i,j,k in vars.keys():
    #    if i == 0 or j == 0:
    #        vars[i,j,k].vtype = GRB.INTEGER
    #        vars[i,j,k].ub = 2

    prize_dict = {
        (i + 1, k): -p  # We need to maximize so negate
        for i, p in enumerate(prize) for k in range(K)
    } 
    depot_dict = {(0,k): 0 for k in range(K)}
    cur_loc_dict = {(len(prize)+1,k):0 for k in range(K)}
    
    delta = m.addVars(prize_dict.keys(), obj=prize_dict, vtype=GRB.BINARY, name='delta') # without depot = node 0
    delta_depot = m.addVars(depot_dict.keys(), vtype=GRB.BINARY, name='delta') 
    delta_cur_loc = m.addVars(cur_loc_dict.keys(), vtype=GRB.BINARY, name='delta')
    # tupledict.sum() 설명
    # x = m.addVars([(1,2), (1,3), (2,3)]) 일때
    # expr = x.sum(1, '*') # LinExpr: x[1,2] + x[1,3]
    # expr = x.sum('*', 3) # LinExpr: x[1,3] + x[2,3]

    # Update and write model
    m.update()
    m.write('dTOP.lp')

    # Write Constraints
    # Add degree-2 constraint (==1 for returning to depot, ==2 * delta for nodes which are not the depot)
    m.addConstrs(vars.sum(i,'*',k) == (1 if i == 0 else 2 * delta[i,k]) for i in range(n-1) for k in range(K))

    # Add degree-2 constraint for cur_loc
    m.addConstrs(vars.sum(len(prize)+1,'*',k) == 1 * delta_cur_loc[len(prize)+1,k] for k in range(K))

    # Limit number of routes / vehicles end at the same depot
    m.addConstr(delta_depot.sum(0,'*') <= K)
    
    # Vehicle start at respective cur_loc 
    m.addConstr(delta_cur_loc.sum() == K)
    
    # Each node visited once 
    m.addConstrs(delta.sum(i+1,'*') <= 1 for i in range(n-2))
    
    # Each route is connected 
    #m.addConstrs(vars.sum('*','*',k) >= (0 if h ==0 else delta[h,k]) for h in range(n+1) for k in range(K)) # required?

    # Length of tour constraint
    #m.addConstr(quicksum(var * dist[i, j] for (i, j), var in vars.items() if j < i) <= max_length)
    m.addConstrs(vars.prod({key:v for key,v in dist.items() if key[2]==k}) <= max_length - cur_tlen[k] for k in range(K))   
    # Update and write model
    m.update()
    m.write('dTOP.lp')
    

    # Optimize model
    m._vars = vars
    m._dvars = delta
    m.Params.lazyConstraints = 1
    m.Params.threads = threads
    if timeout:
        m.Params.timeLimit = timeout
    if gap:
        m.Params.mipGap = gap * 0.01  # Percentage
    
    # optimze model
    m.optimize(subtourelim)
    
    # save model
    x_vals = m.getAttr('x', vars)
    selected = tuplelist((i,j,k) for i,j,k in x_vals.keys() if x_vals[i,j,k] > 0.5)
    # TODO y_vals = m.getAttr(delta)

    tour = subtour(selected, exclude_depot=False)
    #assert tour[0] == 0, "Tour should start with depot"

    return m.objVal, tour