# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:41:54 2023

@author: omlwo
"""


### Cat, Mouse and Cheese. When mouse finds the cat -100, when mouse finds the cheese +100.
## Each time the mouse reaches the cheese or cat, the one he found will change squares.
### Around the cheese is some 'crumbs' +1 point which the mouse can collect one time.

### Each turn the mouse can take 1 of 3 actions.
## Action 1 - Flip a 3-sided coin, if heads more up a square, tails move right, otherwise stay still
## Action 2 - Flip a 3-sided coin, if heads more down a square, tails move left, otherwise stay still

import numpy as np

n = 4 ## rows in grid
m = 3 ## collomns in grid
D= n*m ## number of states
R_Cat = -200
R_Cheese = 100

prob_up = 1/3
prob_right = 1/3
prob_stay_1 = 1 - prob_up - prob_right
prob_left = 1/3
prob_down = 1/3
prob_stay_2 = 1 - prob_up - prob_right

## Will create 2 possible ranges for Cat postion and Cheese position

Cat_Poss = np.array([])
Cheese_Poss = np.array([])
for i in range(D):
    if np.mod(i,n) == 0 or np.mod(i,n) == 1:
        Cat_Poss = np.append(Cat_Poss,i)
    else:
        Cheese_Poss = np.append(Cheese_Poss,i)


gamma = 0.5

# Useful functions to work from grid to vector to grid
def check_grid(n,m,Win,Lose):
    grid = np.zeros([n,m])
    grid[tuple(Win)] = 1
    grid[tuple(Lose)] = -1
    return(grid)
#grid = check_grid(n,m,[0,0],[1,1])
def posgridtovect(X): ## Change the cordindate in the grid to that in a vector. Collumn 1 is 1 to n, collomn 2 n+1 to 2n etc...
    v = n*(X[1]) + X[0]
    return(int(v))
def posvecttogrid(v): ## Change the cordindates in the vector back to that of the grid
    i = np.mod(v,n)
    j = (v - i)/n
    X = [int(i),int(j)]
    return(X)


def neighbours(X): ## returns the vector values of the 4 coridnates around X
    X_up = posgridtovect([int(np.mod(X[0]-1,n)),int(X[1])])
    X_down = posgridtovect([int(np.mod(X[0]+1,n)),int(X[1])])
    X_left = posgridtovect([int(X[0]),int(np.mod(X[1]-1,m))])
    X_right = posgridtovect([int(X[0]),int(np.mod(X[1]+1,m))])
    nbrs = np.array([X_up,X_down,X_left,X_right])
    return(nbrs)

def prob_action(action): ### creates a transition matrix for that actions
    P = np.zeros([D,D])
    if action == 1:
        for i in range(D):
            P[i,i] = prob_stay_1
            X = posvecttogrid(i)
            [X_up,X_down,X_left,X_right] = neighbours(X)
            P[i,X_up] = prob_up
            P[i,X_right] = prob_right
    else:
        for i in range(D):
            P[i,i] = prob_stay_2
            X = posvecttogrid(i)
            [X_up,X_down,X_left,X_right] = neighbours(X)
            P[i,X_down] = prob_down
            P[i,X_left] = prob_down
    return(P)
def prob_a_X_Y(action,X,Y): ## probabilty of moving to Y if you take action in X
    P = prob_action(action)
    i = posgridtovect([int(X[0]),int(X[1])])
    j = posgridtovect([int(Y[0]),int(Y[1])])
    p = P[i,j]
    return(p)
def Return(Cat,Cheese,N_Cat,N_Cheese,X,Y): ##### Calulate the return when moving from X to Y
    Win_V = Cheese[N_Cheese]
    Lose_V = Cat[N_Cat]
    R = np.zeros([n*m,n*m])
    Win = posvecttogrid(int(Win_V))
    Lose = posvecttogrid(int(Lose_V))
    ## identify the four squares that will change because of win
    Win_Neighbour = neighbours(Win)
    Lose_Neighbour = neighbours(Lose)
    for k in range(len(Win_Neighbour)):
        R[Win_Neighbour[k],Win_V] = R_Cheese
    for k in range(len(Lose_Neighbour)):
        R[Lose_Neighbour[k],Lose_V] = R_Cat
    x = posgridtovect(X)
    y = posgridtovect(Y)
    r = R[int(x),int(y)]
    return(r)


##Write a function that given a vector of proabilities to pass at each of the other points, returns a point at random with that probability.

def move(probas):
    cum_sum = np.cumsum(probas)
    p = np.random.random()
    j = 0
    while p>cum_sum[j]:
      j = j + 1
    return(j)


## will add TD_Lambda_Step so that while the MRP is being generated the Value function estimate also


def TD_Lambda_Step(X,R,Y,V,z,lam,alpha):
    delta = R + gamma*V[Y] - V[X]
    for x in range(12):
        z[x] = gamma*lam*z[x]
        if X == x:
            z[x] = 1
        V[x] = V[x] + alpha*delta*z[x]
    return(V,z)
        


def MRP(t,policy,lam,alpha): ### simulates 't' steps of an MRD using policy stated. 
    X = np.empty((t+1,2))
    X_vect = np.empty(t+1)
    R_0 = np.zeros(t)
    X[:] = np.nan
    X_vect[:] = np.nan
    X[0,:] = [np.random.randint(n),np.random.randint(m)]
    X_vect[0] = posgridtovect(X[0,:])
    Cat = np.array([int(np.random.choice(Cat_Poss))]) ### Will put cat only in top half of grid (From 0 to 5)
    Cheese = np.array([int(np.random.choice(Cheese_Poss))]) ### will put cheese only in bottom half of grid (From 6 to 11)
    V_TD_lab = np.zeros([12,t+1])
    z_TD = np.zeros([12,t+1])
    while Cheese == Cat:
        Cheese = np.array([np.random.randint(D)])
    N_Cat = 0
    N_Cheese = 0
    for i in range(t):
        action = policy[int(X[i,0]),int(X[i,1])]
        P = prob_action(int(action))
        v =  move(P[int(X_vect[i]),])
        X_vect[i+1] = v
        X[i+1,:] = posvecttogrid(v)
        R_0[i] = Return(Cat,Cheese,N_Cat,N_Cheese,X[i,:],X[i+1,:])
        V_new,z_new = TD_Lambda_Step(int(X_vect[i]),R_0[i],int(X_vect[i+1]), V_TD_lab[:,i], z_TD[:,i],lam,alpha)
        V_TD_lab[:,i+1] = V_new
        z_TD[:,i+1] = z_new
        if v == Cat[-1]:
            N_Cat = N_Cat + 1
            new_Cat = int(np.random.choice(Cat_Poss))
            while Cheese[-1] == new_Cat:
                new_Cat = int(np.random.choice(Cat_Poss))
            Cat = np.append(Cat,new_Cat)
        if v == Cheese[-1]:
            N_Cheese = N_Cheese + 1
            new_Cheese = int(np.random.choice(Cheese_Poss))
            while Cat[-1] == new_Cheese:
                new_Cheese = int(np.random.choice(Cheese_Poss))
            Cheese = np.append(Cheese,new_Cheese)
    return(X,X_vect,R_0,Cheese,Cat,V_TD_lab,z_TD)

grid_vect = np.zeros([n,m]) ### to understand how the index works
for i in range(n):
    for  j in range(m):
        grid_vect[i,j] = posgridtovect([i,j])   
        
random_policy = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        random_policy[i,j] = np.random.randint(2) + 1

policy_eg_2 = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        a = posgridtovect([i,j]) 
        if a/2 == np.floor(a/2):
            policy_eg_2[i,j] = 1
        else:
            policy_eg_2[i,j] = 2

policy3 = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        a = posgridtovect([i,j]) 
        if np.sum(a == Cat_Poss) == 0:
            policy3[i,j] = 1
        else:
            policy3[i,j] = 2
            
policy1s = np.ones([n,m])            


p_a1 = prob_action(1)
[X,X_vect,reward,Cheese,Cat,V_TD_lab,z_TD] = MRP(100,policy1s,1/2,1/2)
print(np.around(np.reshape(V_TD_lab[:,-1],[4,3],'F')))


### Value Prediction Problems Algorithim 1

def TD0(X_Vect,reward,V_hat_0,gamma,c):
    t = len(X_Vect)
    delta = np.zeros(t)
    V_hat_t = np.zeros([n*m,t])
    V_hat_t[:,0] = V_hat_0
    for i in range(t-1):
        alpha = c/(i+1)
        delta[i+1] = reward[i] + gamma*V_hat_t[int(X_Vect[i+1]),i] - V_hat_t[int(X_Vect[i]),i]
        V_hat_t[:,i+1] = V_hat_t[:,i]
        V_hat_t[int(X_Vect[i]),i+1] = V_hat_t[int(X_Vect[i]),i+1] + (alpha*delta[i+1])
    V_hat = np.zeros([n,m])
    for i in range(n):
        for j in range(m):
            V_hat[i,j] = V_hat_t[posgridtovect([i,j]),-1]
    return([V_hat_t,V_hat,delta])

V_hat_0 = np.zeros(n*m)
[V_hat_t,V_hat,delta] = TD0(X_vect,reward,V_hat_0,gamma,1/2)


V_hat_round = np.around(V_hat)

###  Monte Carlo Method

def everyvisitMC(X_vect,reward,V_0):
    T = len(reward)
    s = 0
    target = np.zeros(12)
    V = np.zeros([12,T+1])
    V[:,0]= V_0
    for i in range(T):
        t = T - 1 - i
        R_t_1 = reward[t]
        X_t = X_vect[t]
        s = R_t_1 + gamma*s
        alpha = 1/2
        target[int(X_t)] = s
        V[:,i+1] = V[:,i]    
        V[int(X_t),i+1] = V[int(X_t),i] + alpha*(target[int(X_t)] - V[int(X_t),i])
    return(V,target)


V_0 = np.zeros(12)
[V,target] = everyvisitMC(X_vect,reward,V_0)
V_end = np.around(np.reshape(V[:,-1],[4,3],'F'))
#print(V_end)

















## Convergence? See if it converges to V. First need to calculate V

## V_ pi is the solution to r_pi = V_pi(I - gamme*P_Pi).
## So first need to define r_pi and P_Pi

## r_pi 
##(r_pi)i = r(x_i,pi(x_i)) = E(R((x_i,pi(x_i))) = 1/3*0(sure there is no reward when staying still) + 1/3*reward(up/down) + 1/3 reward(left or right)
## For the immediate reward left/right or up/down not sure what to use as depends on t. But overall is random so will use (R_Cat+R_Cheese)/(Number of states-1) (A bit like the esperenance)
## This implies its the same for all values

# r_pi_vect = (2/3)*((R_Cat+R_Cheese)/(D-1))*np.ones(D)

# ## P_pi
# ## Martix DxD where row x collomn y is the P(x,y) the prob from moving to x to y. In this case pi(x)
# def P_pi_generate(policy):
#     P_pi = np.zeros([D,D])
#     for i in range(D):
#         P_pi[i,i] = 1/3 
#         pos = posvecttogrid(i)
#         action = policy[int(pos[0]),int(pos[1])]
#         [X_up,X_down,X_left,X_right] = neighbours(pos)       
#         if action == 1:
#             P_pi[i,int(X_up)] = 1/3
#             P_pi[i,int(X_right)] = 1/3
#         else:
#             P_pi[i,int(X_down)] = 1/3
#             P_pi[i,int(X_left)] = 1/3
#     return(P_pi)
            
# P_pi = P_pi_generate(random_policy)

# def V_pi_generate(r_pi,P_pi,gamma):
#     a = np.identity(D) -  gamma*P_pi
#     b = r_pi
#     x = np.linalg.solve(a,b)
#     V_pi = np.reshape(x,[n,m],'F')
#     return(V_pi)

# V_pi = V_pi_generate(r_pi_vect,P_pi,gamma)