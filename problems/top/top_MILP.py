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
    Solves the Euclidan dTOP problem (0112: sequential version) to optimality using the MIP formulation 
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
                #for k in range(K): # always two subtours? TODO CHECK
                k = num_veh
                if tour[0] is not None:
                        model.cbLazy(quicksum(model._vars[i, j, k]
                                              for i, j in itertools.combinations(tour[0], 2))
                                     <= quicksum(model._dvars[i,k] for i in tour[0]) * (len(tour[0]) - 1) / float(len(tour[0])))

                
    # Given a tuplelist of edges, find the shortest subtour
    def subtour(edges_total, exclude_depot=True):
        all_cycle = []
        #for k in range(K):
        k = num_veh
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
      
    tour = []
    objVal = []
    # 0112: Start for loop 
    for num_veh in range(max_vehicle_num):
        
        K = 1 # sequentially computing dtop problem
        
        # Dictionary of Euclidean distance between each pair of points, excluding i=0 (n*(n+1)/2 = 231, n=20+1 per each cur_loc)
        dist = {(i,j,num_veh) :
            math.sqrt(sum((points[num_veh][i][w]-points[num_veh][j][w])**2 for w in range(2)))
            for i in range(n) for j in range(i)}  
        
        if num_veh > 0: # update dist by masking visited nodes 
            # cur_tour[0] = [0, 11, 5, 16, 14, 9, 12, 10, 8, 18, 21]
            # cur_tour[0][1:-1] = [11, 5, 16, 14, 9, 12, 10, 8, 18]
            to_mask = cur_tour[0][1:-1]
            for visited in to_mask:
                # 1. Up to visited (11,0)~(11,10)
                for i in range(visited):    
                    dist[(visited,i,num_veh)] = 1e5
                # 2. (12,11)~(21,11) 
                for i in range(n-1-visited):#0~9
                    dist[(visited+i+1,visited,num_veh)] = 1e5

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
            (i + 1, num_veh): -p  # We need to maximize so negate
            for i, p in enumerate(prize)
        } 
        depot_dict = {(0,num_veh): 0}
        cur_loc_dict = {(len(prize)+1,num_veh):0}
        
        delta = m.addVars(prize_dict.keys(), obj=prize_dict, vtype=GRB.BINARY, name='delta') # without depot = node 0
        delta_depot = m.addVars(depot_dict.keys(), vtype=GRB.BINARY, name='delta') 
        delta_cur_loc = m.addVars(cur_loc_dict.keys(), vtype=GRB.BINARY, name='delta')
        # tupledict.sum() 설명
        # x = m.addVars([(1,2), (1,3), (2,3)]) 일때
        # expr = x.sum(1, '*') # LinExpr: x[1,2] + x[1,3]
        # expr = x.sum('*', 3) # LinExpr: x[1,3] + x[2,3]
    
        # Update and write model
        m.update()
        m.write('dTOP_'+str(num_veh)+'.lp')
    
        # Write Constraints
        # Add degree-2 constraint (==1 for returning to depot, ==2 * delta for nodes which are not the depot)
        m.addConstrs(vars.sum(i,'*', num_veh) == (1 if i == 0 else 2 * delta[i,num_veh]) for i in range(n-1))
    
        # Add degree-2 constraint for cur_loc
        # m.addConstrs(vars.sum(len(prize)+1,'*',k) == 1 * delta_cur_loc[len(prize)+1,k] for k in range(K))
        m.addConstr(vars.sum(len(prize)+1,'*',num_veh) == 1 * delta_cur_loc[len(prize)+1,num_veh])
    
        # Limit number of routes / vehicles end at the same depot
        m.addConstr(delta_depot.sum(0,'*') <= K)
        
        # Vehicle start at respective cur_loc 
        m.addConstr(delta_cur_loc.sum() == K)
        
        # Each node visited once 
        m.addConstrs(delta.sum(i+1,'*') <= 1 for i in range(n-2))
        
        # Each route is connected 
        #m.addConstrs(vars.sum('*','*',k) >= (0 if h ==0 else delta[h,k]) for h in range(n+1) for k in range(K)) # required?
    
        # Length of cur_tour constraint
        #m.addConstrs(vars.prod({key:v for key,v in dist.items() if key[2]==k}) <= max_length - cur_tlen[k] for k in range(K))   
        m.addConstr(vars.prod({key:v for key,v in dist.items() if key[2]==num_veh}) <= max_length - cur_tlen[num_veh])   

        # Update and write model
        m.update()
        m.write('dTOP_'+str(num_veh)+'.lp')
          
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
    
        cur_tour = subtour(selected, exclude_depot=False)
        #assert tour[0] == 0, "Tour should start with depot"
        
        # append cur_tour to tour
        tour.append(cur_tour[0])
        objVal.append(m.objVal)

    return sum(objVal), tour