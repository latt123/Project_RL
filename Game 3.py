# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 16:35:59 2023

@author: omlwo
"""
#### Recreating game. Grid with win square and lose square. Reward for winning +R reward for loosing -L.
#### Two actions, Action 1 - NE. Move north or east with equal prob and Action 2 - SW move south or west with equal proba

import numpy as np

n = 4 ## rows in grid
m = 3 ## collomns in grid
D= n*m ## number of states

gamma = 0.5
Win = [0,1] ## top left of grid
Lose = [2,1] ##bottom left of grid
R_Win = 100 ## Reward for winning
R_Lose = -400 ## Reward for loosing
prob_up = 1/2 ## Probability of moving up with action 1
prob_right = 1/2 ## Probability of moving right with aciton 1
prob_down = 1/2 ## Probability of moving down with aciton 2
prob_left = 1/2 ## Probability of movinf left with action 2 

def check_grid(n,m,Win,Lose):
    grid = np.zeros([n,m])
    grid[tuple(Win)] = 1
    grid[tuple(Lose)] = -1
    return(grid)
grid = check_grid(n,m,Win,Lose)

# Useful functions to work from grid to vector to grid
def posgridtovect(X): ## Collumn 1 is 1 to n, collomn 2 n+1 to 2n etc...
    v = n*(X[1]) + X[0]
    return(v)
def posvecttogrid(v):
    i = np.mod(v,n)
    j = (v - i)/n
    X = [int(i),int(j)]
    return(X)
grid_vect = np.zeros([n,m]) ### to understand how the index works
for i in range(n):
    for  j in range(m):
        grid_vect[i,j] = posgridtovect([i,j])
        

## Matrix de transition of Action 1 and 2
def mat_prob_action1(prob_up,prob_right):
    w = posgridtovect(tuple(Win)) ## index of vector for win square
    l = posgridtovect(tuple(Lose)) ## index of vector for lose square
    p_a1 = np.zeros([D,D])
    if n !=1 and m!= 1:        
        for i in range(D):
            if i != w and i != l:
                X = posvecttogrid(i)
                p_a1[i,posgridtovect([np.mod(X[0]-1,n),X[1]])] = prob_up ## Prob of moving up
                p_a1[i,np.mod(i+n,D)] = prob_right ### Prob of moving right
            else:
               p_a1[w,w] = 1
               p_a1[l,l] = 1
    else:
        if m ==1:
            for i in range(D):
                if i != w and i != l:
                    p_a1[i,np.mod(i-1,D)] = 1
                else:
                    p_a1[w,w] = 1
                    p_a1[l,l] = 1    
        else:
            for i in range(D):
                if i != w and i != l:
                    p_a1[i,np.mod(i+1,D)] = 1
                else:
                    p_a1[w,w] = 1
                    p_a1[l,l] = 1   
    return(p_a1)
def mat_prob_action2(prob_down,prob_left):
    p_a2 = np.zeros([D,D])
    w = posgridtovect(tuple(Win)) ## index of vector for win square
    l = posgridtovect(tuple(Lose)) ## index of vector for lose square
    if n != 1 and m != 1:
        for i in range(D):
            if i != w and i != l:
                X = posvecttogrid(i)
                p_a2[i,posgridtovect([np.mod(X[0]+1,n),X[1]])] = prob_down ## Prob of moving up
                p_a2[i,np.mod(i-n,D)] = prob_left ### Prob of moving right
            else:
                p_a2[w,w] = 1
                p_a2[l,l] = 1
        return(p_a2)
    else:
        if m == 1: 
            for i in range(D):
                if i != w and i != l:
                    p_a2[i,np.mod(i+1,D)] = 1
                else:
                    p_a2[w,w] = 1
                    p_a2[l,l] = 1  
        else:
            for i in range(D):
                if i != w and i != l:
                    p_a2[i,np.mod(i-1,D)] = 1
                else:
                    p_a2[w,w] = 1
                    p_a2[l,l] = 1  
    return(p_a2)
                
                
## The two reward functions (for expectation given an action)
def reward_a1(prob_up,prob_right):
    w = posgridtovect(Win)
    l = posgridtovect(Lose)
    r_a1 = np.zeros([n,m])
    if n != 1 and m!= 1:
        r_a1[np.mod(Win[0]+1,n),Win[1]] = (prob_up)*R_Win
        r_a1[Win[0],np.mod(Win[1]-1,m)] = (prob_right)*R_Win
        r_a1[np.mod(Lose[0]+1,n),Lose[1]] = r_a1[np.mod(Lose[0]+1,n),Lose[1]] +  (prob_up)*R_Lose
        r_a1[Lose[0],np.mod(Lose[1]-1,m)] = r_a1[Lose[0],np.mod(Lose[1]-1,m)] + (prob_right)*R_Lose
    else:
        if n == 1:
            r_a1[0,np.mod(l-1,D)] = R_Lose 
            r_a1[0,np.mod(w-1,D)] = r_a1[0,np.mod(w-1,D)]  + R_Win 
            
        else:
            r_a1[np.mod(w+1,D),0] = R_Win
            r_a1[np.mod(l+1,D),0] = r_a1[np.mod(l+1,D),0] + R_Lose
        r_a1[tuple(Win)] = 0
        r_a1[tuple(Lose)] = 0
    return(r_a1)
def reward_a2(prob_down,prob_left):
    r_a2 = np.zeros([n,m])
    w = posgridtovect(Win)
    l = posgridtovect(Lose)
    if n!=1 and m!=1:
        r_a2[np.mod(Win[0]-1,n),Win[1]] = (prob_down)*R_Win
        r_a2[Win[0],np.mod(Win[1]+1,m)] = (prob_left)*R_Win
        r_a2[np.mod(Lose[0]-1,n),Lose[1]] = r_a2[np.mod(Lose[0]-1,n),Lose[1]] + (prob_down)*R_Lose
        r_a2[Lose[0],np.mod(Lose[1]+1,m)] = r_a2[Lose[0],np.mod(Lose[1]+1,m)] + (prob_left)*R_Lose
    else:
        if n == 1:
            r_a2[0,np.mod(w+1,D)] = R_Win
            r_a2[0,np.mod(l+1,D)] = r_a2[0,np.mod(l+1,D)] + R_Lose
        else:
            r_a2[np.mod(l-1,D),0] = R_Lose    
            r_a2[np.mod(w-1,D),0] = r_a2[np.mod(w-1,D),0] + R_Win  
        r_a2[tuple(Win)] = 0
        r_a2[tuple(Lose)] = 0
    return(r_a2)


p_a1 = mat_prob_action1(prob_up,prob_right)     
p_a2 = mat_prob_action2(prob_down,prob_left)       
r_a1 = reward_a1(prob_up, prob_right)
r_a2 = reward_a2(prob_down, prob_left)
## Finding V_opt

## now iterating with look "n_it" times for V_k and V0 = zeros
def V_opt(V_0,n_it):
    V_0_2 = np.copy(V_0)
    V_k = np.reshape(V_0_2,[D,1],'F')
    for k in range(n_it):
        r_a1_vect = np.reshape(r_a1,[D,1],'F')
        r_a2_vect = np.reshape(r_a2,[D,1],'F')
        T_opt_V_a1 = gamma*np.matmul(p_a1,V_k) + r_a1_vect
        T_opt_V_a2 = gamma*np.matmul(p_a2,V_k) + r_a2_vect
        for i in range(D):
            V_k[i] = max(T_opt_V_a1[i],T_opt_V_a2[i])
        ## for graphic
        # V_k_int = np.reshape(V_k,[n,m],'F')
        # print(V_k_int) #### create a graphic of this to show 'diffusion of the reward as iteration occurs
    V_opt = np.reshape(V_k,[n,m],'F')
    V_opt[tuple(Win)] = R_Win
    V_opt[tuple(Lose)] = R_Lose
    return(V_opt) 

V_0 = np.zeros([n,m])    
V_opt = V_opt(V_0,30)

## Finding the optimal first action for each starting position Q*(x,a)
## Set Q_0 to be zeros

def sup_by_element(vect1,vect2):
    d1 = len(vect1)
    d2 = len(vect2)
    d = min(d1,d2)
    sup = np.zeros([d,1])
    for i in range(d):
        sup[i] = max(vect1[i],vect2[i])
    return(sup)            
def Q_Opt(Q_0,n_it):
    Q_k = Q_0
    for k in range(n_it):
        sup_Q_k = sup_by_element(Q_k[:,0],Q_k[:,1])
        Q_k[:,0] = np.reshape(np.reshape(r_a1,[D,1],'F') + gamma*np.matmul(p_a1,sup_Q_k),D)
        Q_k[:,1] = np.reshape(np.reshape(r_a2,[D,1],'F') + gamma*np.matmul(p_a2,sup_Q_k),D)
    w = posgridtovect(Win)
    l = posgridtovect(Lose)
    Q_k[w,:] = [R_Win,R_Win]
    Q_k[l,:] = [R_Lose,R_Lose]
    return(Q_k)
def preferred_action(Q_opt):
    action_vect = np.zeros([D,1])
    for i in range(D):
        if Q_opt[i,0] > Q_opt[i,1]:
            action_vect[i] = 1
        else:
            if Q_opt[i,0] < Q_opt[i,1]:
                action_vect[i] = 2
            else:
                action_vect[i] = 2
    action_mat = np.reshape(action_vect,[n,m],'F')
    action_mat[tuple(Win)] = 0
    action_mat[tuple(Lose)] = 0
    return(action_mat)

Policy_optimal = preferred_action(Q_Opt(np.zeros([D,2]),30)) ## 

def V_pi(Policy):
    P_pi = np.zeros([D,D])
    for i in range(D):
        X = posvecttogrid(i)        
        if Policy[tuple(X)] == 1:
            P_pi[i,:] = p_a1[i,:]
        else:
            P_pi[i,:] = p_a2[i,:]
    r_pi = np.zeros([D,1])
    for i in range(D):
        X = posvecttogrid(i)        
        if Policy[tuple(X)] == 1:
            r_pi[i,:] = r_a1[tuple(X)]
        else:
            r_pi[i,:] = r_a2[tuple(X)]
    a = np.identity(D) -  gamma*P_pi
    b = r_pi
    x = np.linalg.solve(a,b)
    V_pi = np.reshape(x,[n,m],'F')
    V_pi[tuple(Win)] = R_Win
    V_pi[tuple(Lose)] = R_Lose
    return(V_pi)
V_pi_optimal = V_pi(Policy_optimal)

Policy_random = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        Policy_random[i,j] = np.random.randint(2) + 1
Policy_random[tuple(Win)] = 0
Policy_random[tuple(Lose)] = 0
V_pi_random = V_pi(Policy_random)


## We have our Markov reward process of 'Random Policy' and the game stated.
## Now want to simulate an  ((Xt,Rt+1);t≥0)


##Write a function that given a vector of proabilities to pass at each of the other points, returns a point at random with that probability.

def move(probas):
    cum_sum = np.cumsum(probas)
    p = np.random.random()
    j = 0
    while p>cum_sum[j]:
      j = j + 1
    return(j)

## Function that takes turns policy, and returns a Markov Reward Process

def neighbours(X):
    X_up = posgridtovect([int(np.mod(X[0]-1,n)),int(X[1])])
    X_down = posgridtovect([int(np.mod(X[0]+1,n)),int(X[1])])
    X_left = posgridtovect([int(X[0]),int(np.mod(X[1]-1,m))])
    X_right = posgridtovect([int(X[0]),int(np.mod(X[1]+1,m))])
    nbrs = np.array([X_up,X_down,X_left,X_right])
    return(nbrs)

# ## Define imediate reward for transition between two states
def R(Win,Lose,R_Win,R_Lose,n,m):
    R = np.zeros([n*m,n*m])
    Win_V = posgridtovect(Win)
    Lose_V = posgridtovect(Lose)
    ## identify the four squares that will change because of win
    Win_Neighbour = neighbours(Win)
    Lose_Neighbour = neighbours(Lose)
    for k in range(len(Win_Neighbour)):
        R[Win_Neighbour[k],Win_V] = R_Win
    for k in range(len(Lose_Neighbour)):
        R[Lose_Neighbour[k],Lose_V] = R_Lose
    return(R)
R = R(Win,Lose,R_Win,R_Lose,n,m)


def MRP(turns,policy,R):
    X = np.empty((turns+1,2))
    X_vect = np.empty(turns+1)
    R_0 = np.empty(turns)
    X[:] = np.nan
    R_0[:] = np.nan
    X_vect[:] = np.nan
    X[0,:] = [np.random.randint(n),np.random.randint(m)]
    X_vect[0] = posgridtovect(X[0,:])
    for i in range(turns):
        action = policy[int(X[i,0]),int(X[i,1])]
        if action == 0:
            X[i+1,] = X[i,]
        else:
            vect_point = posgridtovect(X[i,])
            if action == 1:
                probas = p_a1[int(vect_point),]
                v = move(probas)
            else:
                probas = p_a2[int(vect_point),]
                v = move(probas)
            X[i+1,] = posvecttogrid(v)
        X_vect[i+1] = posgridtovect(X[i+1,:])
        R_0[i] = R[int(X_vect[i]),int(X_vect[i+1])]
    t = turns
    MRP = np.empty((t+1,3)) ##Markov Reward Process, collomn 1 is value of t, collomn 2 is Xt, collomn 3 is R(t+1)
    MRP[:] = np.nan
    R_1 = np.append(R_0,0)
    MRP[:,0] = np.linspace(0,t,t+1)
    MRP[:,1] = X_vect    
    MRP[:,2] = R_1  
    return(MRP)

MRP = MRP(10,Policy_random,R)
print(MRP)

A = []
for k in range(10):
    A.append(posvecttogrid(MRP[k,1]))
    


### Value Prediction Problems Algorithim 1

alpha = 1

def TD0(MRP,V_hat_0,gamma,alpha):
    t = len(MRP[:,0])
    delta = np.zeros(t)
    V_hat_t = np.zeros([n*m,t])
    V_hat_t[:,0] = V_hat_0
    for i in range(t-1):
        delta[i+1] = MRP[i,2] + gamma*V_hat_t[int(MRP[i+1,1]),i] - V_hat_t[int(MRP[i,1]),i]
        V_hat_t[:,i+1] = V_hat_t[:,i]
        V_hat_t[int(MRP[i,1]),i+1] = V_hat_t[int(MRP[i,1]),i+1] + alpha*delta[i+1]
    V_hat = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            V_hat[i,j] = V_hat_t[posgridtovect([i,j]),-1]
    return([V_hat_t,V_hat,delta])

V_hat_0 = np.zeros(n*m)
[V_hat_t,V_hat,delta] = TD0(MRP,V_hat_0,gamma,alpha)


