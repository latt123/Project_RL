# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 09:30:57 2023

@author: omlwo
"""
#### ----------- Libaries  --------------------

import numpy as np

#### ----------- Game structure + Comments --------------------
# Second Game
# Again with grid, again with win and lose absorbant square

# This time have two actions
# Action 1, flip '4 sided coin' move in direction dictated. Win 1 coin
# Action 2, move in direction of choise, loose 1 coin (Assume direction of choise is order of priorities:
# 1. Maximise distance from L  2. Minimise distance from Win 3. N 4.S 5.E 6.
# Play 'N' rounds
# If reach W +N and game finished (Stay on Lose)
# If reach L -N and game ends (stay on Lose).

## Comments
# Defining OptStep by 'Not Loosing' Rather than Winning means
# just taking action 2 does not mean you end up at the win square (but does guarentee you don't end up at loose)
# V_ pi not reversible
        
        


#### ----------- Variables  --------------------

#--- Game SetUp------------------
## First define Grid, Win Square (Value Grid +1), Lose Square (Value Grid -1)
n = 3  ## number of rows in grid
m = n  ## number of colloms in grid (nxn in this case)
W_i = int(np.floor(n/4)) ## Row of winning square
W_j = int(np.floor(n/4)) ## Collomn of winning square
L_i = 2 #Row of loosing square
L_j = 2 #Collomn of loosing square
# L_i = int(3*np.floor(n/4)) #alternative
# L_j = int(3*np.floor(n/4)) #alternative
grid = np.zeros([n,m])
grid[W_i,W_j] = 1 ## grid to visulaise Win and loose in right place
grid[L_i,L_j] = -1

## Similarly define matrix of distance from loosing and winning square (useful later)
dist_W = np.zeros([n,m])
for i in range(n): ## defining dist_W
    for j in range(m):
        dist_W[i,j] = min(abs(W_i - i),m - abs(W_i - i)) + min(abs(W_j - j),n - abs(W_j - j)) ##min and abs to deal with fact that grid is circular
dist_L = np.zeros([n,m])
for i in range(n): ## defining dist_L
    for j in range(m):
        dist_L[i,j] = min(abs(L_i - i),m - abs(L_i - i)) + min(abs(L_j - j),n - abs(L_j - j))
        

# -------Example Policy and Start/Turns -----------      

#Consider the policy where you deicide 50/50 each turn which action to take
## Proba 'p_a1' of taking action1 in each state
p_a1 = 1/2
Policy  = p_a1*np.ones([n,m]) ## at each state the prob of action 1 is value of matrix (in this case not dependant on time)
Start = [np.floor(n/2), np.floor(n/2)] ## this case chosen from middle
Turns = 10

#### ----------- Functions to play game  --------------------

def action1(X):  ###Input state, output new state after action
    if np.all(X == [W_i,W_j]):
        Y = [W_i,W_j]
    else:
        if np.all(X == [L_i,L_j]):
            Y = [L_i,L_j]
        else:
            D = np.random.randint(4) ### direction to take, (if 0 - N, 1- E, 2 - S, 3 - W)
            Y = np.copy(X)
            if D == 0:
                Y[0] = np.mod(X[0]-1,n)
            else:
                if D == 1:
                    Y[1] = np.mod(X[1]+1,m)
                else:
                    if D == 2:
                        Y[0] = np.mod(X[0]+1,n)    
                    else:
                        Y[1] = np.mod(X[1]-1,m)
    Y = [int(Y[0]),int(Y[1])]
    return(Y)

def genDir(X):       ## To Set up action2
    Dir = np.zeros([4,3])
    X[0] = int(X[0])
    X[1] = int(X[1]) ### input Dir matrix, the 1st collomn is distance from loosing if move NESW. 2nd dist from winning
    Dir[0,:] = [dist_L[np.mod(X[0]-1,n),X[1]],dist_W[np.mod(X[0]-1,n),X[1]],0] ## distance from loosing if move north
    Dir[1,:] = [dist_L[X[0],np.mod(X[1]+1,m)],dist_W[X[0],np.mod(X[1]+1,m)],1] ## distance from loosing if move east
    Dir[2,:] = [dist_L[np.mod(X[0]+ 1,n),X[1]],dist_W[np.mod(X[0]+ 1,n),X[1]],2] ## distance from loosing if move south
    Dir[3,:] = [dist_L[X[0],np.mod(X[1]-1,m)],dist_W[X[0],np.mod(X[1]-1,m)],3] ## distance from loosing if move west               
    return(Dir)
     
def optstep(Dir): ###  Set up action 2 toinput Dir matrix, the 1st collomn is distance from loosing if move NESW. 2nd dist from winning
    max_L = max(Dir[:,0])
    max_Dir_values = [Dir[:,0] == max_L]
    Dir2 = Dir[tuple(max_Dir_values)]
    if len(Dir2) == 1:
        otpD =  Dir2[0,2] ### if only one that is best for loose choose this one
    else:
        min_W = min(Dir2[:,1])
        min_Dir2_values = [Dir2[:,1] == min_W]
        Dir3 = Dir2[tuple(min_Dir2_values)]
        otpD = Dir3[0,2] ## if only 1 value, this is index, if more than one then take first one anyway
    return(otpD)
                 
def action2(X):  ###Input state, output new state after action
    if np.all(X == [W_i,W_j]):
        Y = [W_i,W_j]
    else:
        if np.all(X == [L_i,L_j]):
            Y = [L_i,L_j]
        else:    
            X = [int(X[0]),int(X[1])]
            Dir = genDir(X)
            otpD = optstep(Dir)
            if otpD == 0:
                Y = [np.mod(X[0]-1,n),X[1]]
            else:
                if otpD == 1:
                    Y = [X[0],np.mod(X[1]+1,m)]
                else:
                    if otpD == 2:
                        Y = [np.mod(X[0]+1,n),X[1]]
                    else:
                        Y = [X[0],np.mod(X[1]-1,m)]
    Y = [int(Y[0]),int(Y[1])]
    return(Y)

##First define r_a1 and r_a2 ##REward for action 1 and 2 in each state 
R_W = 10
R_L = -10
r_a1 = np.ones([n,m])
r_a1[W_i,W_j] = R_W
r_a1[L_i,L_j] = R_L
r_a2 = (-1)*np.ones([n,m])
r_a2[W_i,W_j] = R_W
r_a2[L_i,L_j] = R_L


def playgame(Start,Turns,Policy,r_a1,r_a2):
    N = Turns
    X = np.zeros([N+1,2]) ## The path of the player
    R = np.zeros([N+1,1]) ## Row t is total money at this point
    R[0] = 0
    X[0,:] = [int(Start[0]),int(Start[1])]
    for k in range(N):
        X_pol = Policy[int(X[k,0]),int(X[k,1])]
        a = np.random.rand(1)
        if a < X_pol:
            X[k+1,:] = action1(X[k,:])
            R[k+1] = R[k] + r_a1[int(X[k+1,0]),int(X[k+1,1])] ## gain 1  coin by taking chance
        else:### in the case of action 2
            X[k+1,:] = action2(X[k,:])
            R[k+1] = R[k] + r_a2[int(X[k+1,0]),int(X[k+1,1])] ## loose 1  coin by choosing best step to take
    return([X,R])
         
def gameresult(Start,Turns,Policy,r_a1,r_a2):
    [Path,Reward] = playgame(Start,Turns,Policy,r_a1,r_a2)
    Final_Position = Path[-1]
    Final_Score = int(Reward[-1])
    if np.all(Final_Position == [W_i,W_j]):
        print("You Won, Final Score",Final_Score)
    else:
        if np.all(Final_Position == [L_i,L_j]):
            print("You Lose, Final Score",Final_Score)
        else:
            print("You didn't win or lose, Final Score",Final_Score)  

#### ----------- Calculating value function V_pi for policy pi --------------------

## Finding V_Pi. Solution 'x' of lin equation Ax = b with A = P_pi*gamma + I, b = r_pi
gamma = 1.5

# Useful functions to work from grid to vector to grid
def posgridtovect(X): ## Collumn 1 is 1 to n, collomn 2 n+1 to 2n etc...
    v = n*(X[1]) + X[0]
    return(v)
def posvecttogrid(v):
    i = np.mod(v,n)
    j = (v - i)/n
    X = [int(i),int(j)]
    return(X)

# P_pi (n*m)x(n*m) matrix of proba 
D = (n*m)
P_pi = np.zeros([D,D])
for i in range(D): ## defining P_pi
        P = np.copy(P_pi)
        X = posvecttogrid(i)
        opt_D = action2(X)
        j_opt = posgridtovect(opt_D)
        P_pi[i,np.mod(i+n,D)] = 1/4*p_a1
        P_pi[i,np.mod(i-n,D)] = 1/4*p_a1
        P_pi[i,posgridtovect([np.mod(X[0]-1,n),X[1]])] = 1/4*p_a1
        P_pi[i,posgridtovect([np.mod(X[0]+1,n),X[1]])] = 1/4*p_a1
        P_pi[i,j_opt] = P_pi[i,np.mod(j_opt,D)]  + (1-p_a1)
w = posgridtovect([W_i,W_j])
l = posgridtovect([L_i,L_j])
P_pi[w,:] = np.zeros([1,D])
P_pi[w,w] = 1
P_pi[l,:] = np.zeros([1,D])
P_pi[l,l] = 1

### Now define r_pi
## Expectation of reward is 0 (1/2 for +1 1/2 for -1 at every state except Winning and Loosing)
r_pi = np.zeros([D,1])
r_pi[w] = R_W
r_pi[l] = R_L

## Now solve r_pi = (gamma*P_pi - I)*V_pi to find V_pi
a = np.identity(D) -  gamma*P_pi
b = r_pi

x = np.linalg.solve(a,b)
V_pi = np.reshape(x,[n,m],'F') ### Transform back into grid

#### ----------- Calculating value function V*, optimal value function--------------------

## Iterating 'n_it' times to find V*    
def approx_V_opt(V_0,n_it):
    V_k = V_0
    for k in range(n_it): ## generating V_k
        V_k_b = np.copy(V_k)
        for i in range(n):
            for j in range(m):
                sum_P_V_a1 = (1/4)*(V_k_b[np.mod(i-1,n),j] + V_k_b[i,np.mod(j+1,m)] + V_k_b[np.mod(i+1,n),j] + V_k_b[i,np.mod(j-1,m)])
                sum_P_V_a2 = V_k_b[action2([i,j])[0],action2([i,j])[1]]
                V_k[i,j] = max(r_a1[i,j] + gamma*sum_P_V_a1,r_a2[i,j]+gamma*sum_P_V_a2)
        V_k[W_i,W_j] = max(r_a1[W_i,W_j],r_a2[W_i,W_j])
        V_k[L_i,L_j] = max(r_a1[L_i,L_j],r_a2[L_i,L_j])
    V_opt = V_k
    return(V_opt)
V_0 = np.zeros([n,m])
V_opt = approx_V_opt(V_0,10)



        

