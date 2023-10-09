# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 13:00:01 2023

@author: omlwo
"""

import numpy as np
import random

# Useful functions to work from grid to vector to grid
def posgridtovect(X): ## Collumn 1 is 1 to n, collomn 2 n+1 to 2n etc...
    v = n*(X[1]) + X[0]
    return(v)
def posvecttogrid(v):
    i = np.mod(v,n)
    j = (v - i)/n
    X = [int(i),int(j)]
    return(X)

## First define useful matrices and constants

n = 4  ## number of rows in grid
m = 3  ## number of colloms in grid (nxn in this case)

## Matrix of distances for winning
## First define coordinates of winning
W_i = 2 ## (top right corner of grid)
W_j = 1 ## (top right corner of grid)
Win = np.array([W_i,W_j])
R_Win = 100
dist_W = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        dist_W[i,j] = min(abs(W_i - i),m - abs(W_i - i)) + min(abs(W_j - j),n - abs(W_j - j)) ##min and abs to deal with fact that grid is circular
        
## Similarly define cordinates of winning, and matrix of distance from loosing square

L_i = 0
L_j = 2
Lose = np.array([L_i,L_j])
R_Lose = -200
dist_L = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        dist_L[i,j] = min(abs(L_i - i),m - abs(L_i - i)) + min(abs(L_j - j),n - abs(L_j - j))


grid = np.zeros([n,m])
grid[W_i,W_j] = R_Win
grid[L_i,L_j] = R_Lose

reward_vect = np.zeros([n*m])
reward_vect[int(posgridtovect([int(W_i),int(W_j)]))] = R_Win
reward_vect[int(posgridtovect([int(L_i),int(L_j)]))] = R_Lose


policy1s = np.ones([n,m]) ## Always choose action 1
policy2s = np.zeros([n,m]) ## Always choose action 2
## Matrice of expected rewards for action a_1
## Action a1 is to move up or to the right. a2 to left or down. For now with equal probability. Prob given below

p_up_a1 = 0.5
p_right_a1 = 0.5
p_down_a2 = 0.5
p_left_a2 = 0.5

## From this calculate expected reward r(x,a) for a1 and a2
## Set reward to winning equal to R
R = 10
gamma = 1/2


r_a1 = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        r_a1_W = p_up_a1*(dist_W[i,j]-dist_W[np.mod(i-1,n),j]) + p_right_a1*(dist_W[i,j]-dist_W[i,np.mod(j+1,m)])
        r_a1_L = p_up_a1*(dist_L[i,j]-dist_L[np.mod(i-1,n),j]) + p_right_a1*(dist_L[i,j]-dist_L[i,np.mod(j+1,m)])
        r_a1[i,j] = r_a1_W - r_a1_L
r_a1[W_i,W_j] = R
r_a1[L_i,L_j] = -R


r_a2 = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        r_a2_W = p_down_a2*(dist_W[i,j]-dist_W[np.mod(i+1,n),j]) + p_left_a2*(dist_W[i,j]-dist_W[i,np.mod(j-1,m)])
        r_a2_L = p_down_a2*(dist_L[i,j]-dist_L[np.mod(i+1,n),j]) + p_left_a2*(dist_L[i,j]-dist_L[i,np.mod(j-1,m)])
        r_a2[i,j] = r_a2_W - r_a2_L
r_a2[W_i,W_j] = R
r_a2[L_i,L_j] = -R
        
## iterating V "by hand"

V_0 = np.zeros([n,m])
V_1 = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        V_1[i,j] = max(r_a1[i,j],r_a2[i,j])

        
V_2 = np.zeros([n,m])
for i in range(n):
    for j in range(m):
        sum_P_V_a1 = p_up_a1*V_1[np.mod(i-1,n),j] + p_right_a1*V_1[i,np.mod(j+1,m)]
        sum_P_V_a2 = p_down_a2*V_1[np.mod(i+1,n),j] + p_left_a2*V_1[i,np.mod(j-1,m)]
        V_2[i,j] = max(r_a1[i,j] + gamma*sum_P_V_a1,r_a2[i,j]+gamma*sum_P_V_a2)
V_2[W_i,W_j] = 2*max(r_a1[W_i,W_j],r_a2[W_i,W_j])
V_2[L_i,L_j] = 2*max(r_a1[L_i,L_j],r_a2[L_i,L_j])
        
## now iterating with look "n_it" times for V_k and V0 = zeros
n_it =50 ## number of iterations
V_0 = np.zeros([n,m])
V_k = V_0
for k in range(n_it):
    V_k_b = np.copy(V_k)
    for i in range(n):
        for j in range(m):
            sum_P_V_a1 = p_up_a1*V_k_b[np.mod(i-1,n),j] + p_right_a1*V_k_b[i,np.mod(j+1,m)]
            sum_P_V_a2 = p_down_a2*V_k_b[np.mod(i+1,n),j] + p_left_a2*V_k_b[i,np.mod(j-1,m)]
            V_k[i,j] = max(r_a1[i,j] + gamma*sum_P_V_a1,r_a2[i,j]+gamma*sum_P_V_a2)
    V_k[W_i,W_j] = (k+1)*max(r_a1[W_i,W_j],r_a2[W_i,W_j])
    V_k[L_i,L_j] = (k+1)*max(r_a1[L_i,L_j],r_a2[L_i,L_j])
#print(V_k)
   


#Consider the policy where you deicide 50/50 each turn which action to take
## Proba 'p_a1' of taking action1 in each state
p_a1 = 1/2
Policy  = p_a1*np.ones([n,m]) ## at each state the prob of action 1 is value of matrix (in this case not dependant on time)


# P_pi (n*m)x(n*m) matrix of proba 
D = (n*m)
P_pi = np.zeros([D,D])
for i in range(D): ## defining P_pi
    P = np.copy(P_pi)
    X = posvecttogrid(i)
    P_pi[i,np.mod(i+n,D)] = 1/2*Policy[tuple(X)] ### moving East
    P_pi[i,posgridtovect([np.mod(X[0]-1,n),X[1]])] = 1/2*Policy[tuple(X)] ### Moving North
    P_pi[i,np.mod(i-n,D)] = 1/2*(1-Policy[tuple(X)]) ### moving West
    P_pi[i,posgridtovect([np.mod(X[0]+1,n),X[1]])] = 1/2*(1-Policy[tuple(X)]) ### Moving South
w = posgridtovect([W_i,W_j])
l = posgridtovect([L_i,L_j])
P_pi[w,:] = np.zeros([1,D])
P_pi[w,w] = 1
P_pi[l,:] = np.zeros([1,D])
P_pi[l,l] = 1

### Now define r_pi
## Expectation of reward is 0 (1/2 for +1 1/2 for -1 at every state except Winning and Loosing)
r_pi = np.zeros([D,1])
for i in range(D):
    X = posvecttogrid(i)
    r_pi[i] = Policy[tuple(X)]*r_a1[tuple(X)] + (1-Policy[tuple(X)])*r_a2[tuple(X)]
    

## Now solve r_pi = (gamma*P_pi - I)*V_pi to find V_pi
a = np.identity(D) -  gamma*P_pi
b = r_pi

x = np.linalg.solve(a,b)
V_pi = np.reshape(x,[n,m],'F') ### Transform back into grid

policy_2 = np.zeros([n,m]) ### Define each policy as the probability to take action 1 that square
for i in range(n):
    for j in range(m):
        a = posgridtovect([i,j]) 
        if a/2 == np.floor(a/2):
            policy_2[i,j] = 1
        else:
            policy_2[i,j] = 0
            
def neighbours(X): ## returns the vector values of the 4 coridnates around X
    X_up = posgridtovect([int(np.mod(X[0]-1,n)),int(X[1])])
    X_down = posgridtovect([int(np.mod(X[0]+1,n)),int(X[1])])
    X_left = posgridtovect([int(X[0]),int(np.mod(X[1]-1,m))])
    X_right = posgridtovect([int(X[0]),int(np.mod(X[1]+1,m))])
    nbrs = np.array([X_up,X_down,X_left,X_right])
    return(nbrs)
            
def MRP(turns,policy,X_0_vect,Win,Lose):
    X = np.array([X_0_vect])
    R = np.zeros(turns)
    Win_Vect = posgridtovect([int(Win[0]),int(Win[1])])
    Lose_Vect = posgridtovect([int(Lose[0]),int(Lose[1])])
    for i in range(turns):
        pos = int(X[-1])
        pos_grid = posvecttogrid(pos)
        if pos == Win_Vect or pos == Lose_Vect:
            X = np.append(X,X[-1])
            R[i] = 0        
        else:
            [pos_up,pos_down,pos_left,pos_right] = neighbours([int(pos_grid[0]),int(pos_grid[1])])
            p_a1 = policy[int(pos_grid[0]),int(pos_grid[1])]
            dice = np.random.random()
            if dice < p_a1:
                coin = np.random.random()
                if coin < 1/2:
                    X = np.append(X,pos_up)
                else:
                    X = np.append(X,pos_right)
            else:
                coin = np.random.random()
                if coin < 1/2:
                    X = np.append(X,pos_down)
                else:
                    X = np.append(X,pos_left)
            if X[-1] == Win_Vect:
                R[i] = R_Win
            else:
                if X[-1] == Lose_Vect:
                    R[i] = R_Lose
                else:
                    R[i] = 0
    return(X,R)
    

[X_eg2,R_eg2] = MRP(10,policy_2,5,Win,Lose)

def find_r_pi(policy):
    r_pi = np.zeros(n*m)
    for i in range(n*m):
        pos = int(i)
        pos_grid = posvecttogrid(pos)
        [pos_up,pos_down,pos_left,pos_right] = neighbours([int(pos_grid[0]),int(pos_grid[1])])
        pol = policy[int(pos_grid[0]),int(pos_grid[1])]
        if pol == 1:
            r_pi[i] = 1/2*reward_vect[int(pos_up)] + 1/2*reward_vect[int(pos_right)]
        else:
            r_pi[i] = 1/2*reward_vect[int(pos_down)] + 1/2*reward_vect[int(pos_left)]
    Win_Vect = posgridtovect([int(Win[0]),int(Win[1])])
    Lose_Vect = posgridtovect([int(Lose[0]),int(Lose[1])])
    r_pi[int(Win_Vect)] = R_Win
    r_pi[int(Lose_Vect)] = R_Lose
    return(r_pi)            
r_pi = find_r_pi(policy1s)


def find_P_pi(policy):
    P_pi = np.zeros([n*m,n*m])
    for i in range (n*m):
        pos = int(i)
        pos_grid = posvecttogrid(pos)
        [pos_up,pos_down,pos_left,pos_right] = neighbours([int(pos_grid[0]),int(pos_grid[1])])
        pol = policy[int(pos_grid[0]),int(pos_grid[1])]
        if pol == 1:
            P_pi[pos,int(pos_right)] = 1/2
            P_pi[pos,int(pos_up)] = 1/2
        else:
            P_pi[pos,int(pos_down)] = 1/2
            P_pi[pos,int(pos_left)] = 1/2
    Win_Vect = posgridtovect([int(Win[0]),int(Win[1])])
    Lose_Vect = posgridtovect([int(Lose[0]),int(Lose[1])])
    P_pi[int(Win_Vect),:] = np.zeros(n*m)
    P_pi[int(Lose_Vect),:] = np.zeros(n*m)
    P_pi[int(Win_Vect),int(Win_Vect)] = 1
    P_pi[int(Lose_Vect),int(Lose_Vect)] = 1
    return(P_pi)

P_pi = find_P_pi(policy1s)
gamma = 1/2
V_pi = np.matmul(np.linalg.inv(np.eye(n*m)-gamma*P_pi),r_pi)



## Finding the optimal first action for each starting position Q*(x,a)
## Set Q_0 to be zeros



# Policy_optimal = preferred_action(Q_Opt(np.zeros([D,2]),30)) ##



r_a1_vect = find_r_pi(policy1s).reshape([12,1])
r_a2_vect = find_r_pi(policy2s).reshape([12,1])

P_x_a1_y = find_P_pi(policy1s)
P_x_a2_y = find_P_pi(policy2s)

Q_0 = np.zeros([12,2])
# sup_Q_0 = sup_by_element(Q_0[:,0],Q_0[:,1])
# Q_1 = np.zeros([12,2])
# Q_1[:,0] = (r_a1_vect + gamma*np.matmul(P_x_a1_y,sup_Q_0)).reshape(12)
# Q_1[:,1] = (r_a2_vect + gamma*np.matmul(P_x_a2_y,sup_Q_0)).reshape(12)



def sup_by_element(vect1,vect2):
    d1 = len(vect1)
    d2 = len(vect2)
    d = min(d1,d2)
    sup = np.zeros([d,1])
    for i in range(d):
        sup[i] = max(vect1[i],vect2[i])
    return(sup)    


# Finding the optimal first action for each starting position Q*(x,a)
# Set Q_0 to be zeros        
def Q_Opt(Q_0,n_it):
    Q_k = Q_0
    V_k = np.zeros([D,1])
    V_it =  np.zeros([D,1])
    for k in range(n_it):
        sup_Q_k = sup_by_element(Q_k[:,0],Q_k[:,1])
        Q_k[:,0] = np.reshape(r_a1_vect + gamma*np.matmul(P_x_a1_y,sup_Q_k),D)
        Q_k[:,1] = np.reshape(r_a2_vect + gamma*np.matmul(P_x_a2_y,sup_Q_k),D)
        for i in range(D):
            V_k[i,0] = max(Q_k[i,0],Q_k[i,1])
        V_it = np.concatenate((V_it,V_k))
    w = int(posgridtovect(Win))
    l = int(posgridtovect(Lose))
    #Q_k[w,:] = [R_Win,R_Win]
    #Q_k[l,:] = [R_Lose,R_Lose] 
    V_it = np.reshape(V_it,[D,(n_it+1)],'F')
    return([Q_k,V_k,V_it])

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


[Q_k,V_k,V_it] =  Q_Opt(Q_0,35)

V_0 = np.zeros([n,m])
for k in range(D):
    [i,j] = posvecttogrid(int(k))
    V_0[int(i),int(j)] = V_it[k,0]
V_1 = np.zeros([n,m])
for k in range(D):
    [i,j] = posvecttogrid(int(k))
    V_1[int(i),int(j)] = V_it[k,1]
V_2 = np.zeros([n,m])
for k in range(D):
    [i,j] = posvecttogrid(int(k))
    V_2[int(i),int(j)] = V_it[k,2]   
V_et = np.zeros([n,m])
for k in range(D):
    [i,j] = posvecttogrid(int(k))
    V_et[int(i),int(j)] = V_it[k,-1] 
##print(V_et)



policy_optimal = np.zeros([n,m])
policy_optimal[0,0] = 1   #0
policy_optimal[1,0] = 0   #1
policy_optimal[2,0] = 1   #2
policy_optimal[3,0] = 1   #3
policy_optimal[0,1] = 0   #4
policy_optimal[1,1] = 0   #5
policy_optimal[2,1] = 1   #6
policy_optimal[3,1] = 1   #7
policy_optimal[0,2] = 1   #8
policy_optimal[1,2] = 0   #9
policy_optimal[2,2] = 0   #10
policy_optimal[3,2] = 1   #11
P_pi_opitmal = find_P_pi(policy_optimal)
r_pi_optimal = find_r_pi(policy_optimal)

V_pi = np.reshape(np.matmul(np.linalg.inv(np.eye(n*m)-gamma*P_pi_opitmal),r_pi_optimal),[4,3],'F')
##print(V_pi)

def MRP_MC(start,policy,Win,Lose): ## vector inputs for win lose start
    X = np.array([start])
    while X[-1] != Win and X[-1] != Lose:
        pos_vect = int(X[-1])
        pos_grid = posvecttogrid(pos_vect)
        pol = policy[int(pos_grid[0]),int(pos_grid[1])]
        [X_up,X_down,X_left,X_right] = neighbours(pos_grid) ## gives vector position
        if pol == 1:
            if random.random() < 1/2:
                X = np.append(X,X_up)
            else:
                X = np.append(X,X_right)
        else:
            if random.random() < 1/2:
                X = np.append(X,X_down)
            else:
                X = np.append(X,X_left)
    length_X = len(X)
    if X[-1] == Win:
        R = gamma**(length_X -1)*R_Win
    else:
        R = gamma**(length_X -1)*R_Lose
    return([X,R,length_X])
            

Win_Vect = int(posgridtovect([int(W_i),int(W_j)]))
Lose_Vect = int(posgridtovect([int(L_i),int(L_j)]))   

[X,R,length_X] = MRP_MC(3,policy1s,Win_Vect,Lose_Vect)



def montecarlo(n_episodes,policy,Win_Vect,Lose_Vect):
    R_mat = np.zeros([12,n_episodes])
    for i in range(12):
        for j in range(n_episodes):
            start = int(i)
            [X,R_j,length_X] = MRP_MC(start,policy,Win_Vect,Lose_Vect)
            R_mat[i,j] = R_j
    return(R_mat)


n_episodes = 500
R_MC = montecarlo(n_episodes,policy_optimal,Win_Vect,Lose_Vect)
            
V = np.reshape(np.mean(R_MC,axis=1),[4,3],'F')
V_round = np.around(V)
print(V_round)     


for i in range(5):
    [X,R,length_X] = MRP_MC(0,policy1s,Win_Vect,Lose_Vect)
    print("X is", X,"R is ",R)
    print("--------------------")
