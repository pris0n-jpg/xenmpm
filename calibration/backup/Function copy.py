# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:04:56 2022

@author: dell
"""

import numpy as np
import scipy as sp
from scipy.sparse.linalg import spsolve
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#import cv2

'''
#8节点单元形函数
def FN(a,b,c,N):
    N[5] = 0.125*(1-a)*(1-b)*(1-c)
    N[4] = 0.125*(1+a)*(1-b)*(1-c)
    N[7] = 0.125*(1+a)*(1+b)*(1-c)
    N[6] = 0.125*(1-a)*(1+b)*(1-c)
    N[1] = 0.125*(1-a)*(1-b)*(1+c)
    N[0] = 0.125*(1+a)*(1-b)*(1+c)
    N[3] = 0.125*(1+a)*(1+b)*(1+c)
    N[2] = 0.125*(1-a)*(1+b)*(1+c)

#8节点单元形函数j局部坐标偏导a
def FNA(b,c,NA):
    NA[5] = -0.125*(1-b)*(1-c)
    NA[4] = 0.125*(1-b)*(1-c)
    NA[7] = 0.125*(1+b)*(1-c)
    NA[6] = -0.125*(1+b)*(1-c)
    NA[1] = -0.125*(1-b)*(1+c)
    NA[0] = 0.125*(1-b)*(1+c)
    NA[3] = 0.125*(1+b)*(1+c)
    NA[2] = -0.125*(1+b)*(1+c)

#8节点单元形函数j局部坐标偏导b
def FNB(a,c,NB):
    NB[5] = -0.125*(1-a)*(1-c)
    NB[4] = -0.125*(1+a)*(1-c)
    NB[7] = 0.125*(1+a)*(1-c)
    NB[6] = 0.125*(1-a)*(1-c)
    NB[1] = -0.125*(1-a)*(1+c)
    NB[0] = -0.125*(1+a)*(1+c)
    NB[3] = 0.125*(1+a)*(1+c)
    NB[2] = 0.125*(1-a)*(1+c)

#8节点单元形函数j局部坐标偏导c
def FNC(a,b,NC):
    NC[5] = -0.125*(1-a)*(1-b)
    NC[4] = -0.125*(1+a)*(1-b)
    NC[7] = -0.125*(1+a)*(1+b)
    NC[6] = -0.125*(1-a)*(1+b)
    NC[1] = 0.125*(1-a)*(1-b)
    NC[0] = 0.125*(1+a)*(1-b)
    NC[3] = 0.125*(1+a)*(1+b)
    NC[2] = 0.125*(1-a)*(1+b)
'''

# node_seq = [1,2,3,0,5,6,7,4]
node_seq = [1,5,6,2,0,4,7,3]

#8节点单元形函数
def FN(a,b,c,N):
    N[node_seq[0]] = 0.125*(1-a)*(1-b)*(1-c)
    N[node_seq[1]] = 0.125*(1+a)*(1-b)*(1-c)
    N[node_seq[2]] = 0.125*(1+a)*(1+b)*(1-c)
    N[node_seq[3]] = 0.125*(1-a)*(1+b)*(1-c)
    N[node_seq[4]] = 0.125*(1-a)*(1-b)*(1+c)
    N[node_seq[5]] = 0.125*(1+a)*(1-b)*(1+c)
    N[node_seq[6]] = 0.125*(1+a)*(1+b)*(1+c)
    N[node_seq[7]] = 0.125*(1-a)*(1+b)*(1+c)

#8节点单元形函数j局部坐标偏导a
def FNA(b,c,NA):
    NA[node_seq[0]] = -0.125*(1-b)*(1-c)
    NA[node_seq[1]] = 0.125*(1-b)*(1-c)
    NA[node_seq[2]] = 0.125*(1+b)*(1-c)
    NA[node_seq[3]] = -0.125*(1+b)*(1-c)
    NA[node_seq[4]] = -0.125*(1-b)*(1+c)
    NA[node_seq[5]] = 0.125*(1-b)*(1+c)
    NA[node_seq[6]] = 0.125*(1+b)*(1+c)
    NA[node_seq[7]] = -0.125*(1+b)*(1+c)

#8节点单元形函数j局部坐标偏导b
def FNB(a,c,NB):
    NB[node_seq[0]] = -0.125*(1-a)*(1-c)
    NB[node_seq[1]] = -0.125*(1+a)*(1-c)
    NB[node_seq[2]] = 0.125*(1+a)*(1-c)
    NB[node_seq[3]] = 0.125*(1-a)*(1-c)
    NB[node_seq[4]] = -0.125*(1-a)*(1+c)
    NB[node_seq[5]] = -0.125*(1+a)*(1+c)
    NB[node_seq[6]] = 0.125*(1+a)*(1+c)
    NB[node_seq[7]] = 0.125*(1-a)*(1+c)

#8节点单元形函数j局部坐标偏导c
def FNC(a,b,NC):
    NC[node_seq[0]] = -0.125*(1-a)*(1-b)
    NC[node_seq[1]] = -0.125*(1+a)*(1-b)
    NC[node_seq[2]] = -0.125*(1+a)*(1+b)
    NC[node_seq[3]] = -0.125*(1-a)*(1+b)
    NC[node_seq[4]] = 0.125*(1-a)*(1-b)
    NC[node_seq[5]] = 0.125*(1+a)*(1-b)
    NC[node_seq[6]] = 0.125*(1+a)*(1+b)
    NC[node_seq[7]] = 0.125*(1-a)*(1+b)


#坐标变换,得到Nx.Ny.Nz,xo=XO[Ele[i,j]-1]
def jaco(a,b,c,xo,yo,zo):
    NA = np.zeros(8)
    NB = np.zeros(8)
    NC = np.zeros(8)
    FNA(b,c,NA)
    FNB(a,c,NB)
    FNC(a,b,NC)
    J = np.zeros((3,3))
    for i in range(8):
        J[0,0] += NA[i]*xo[i]
        J[0,1] += NA[i]*yo[i]
        J[0,2] += NA[i]*zo[i]
        J[1,0] += NB[i]*xo[i]
        J[1,1] += NB[i]*yo[i]
        J[1,2] += NB[i]*zo[i]
        J[2,0] += NC[i]*xo[i]
        J[2,1] += NC[i]*yo[i]
        J[2,2] += NC[i]*zo[i]
    det_J=np.linalg.det(J)
    if det_J <= 0:
        print('雅克比行列式<0')
        exit()
    J1 = np.linalg.inv(J)
    Nx = np.zeros(8)
    Ny = np.zeros(8)
    Nz = np.zeros(8)
    for i in range(8):
        Nx[i] = J1[0,0]*NA[i] + J1[0,1]*NB[i] + J1[0,2]*NC[i]
        Ny[i] = J1[1,0]*NA[i] + J1[1,1]*NB[i] + J1[1,2]*NC[i]
        Nz[i] = J1[2,0]*NA[i] + J1[2,1]*NB[i] + J1[2,2]*NC[i]
    return det_J,Nx,Ny,Nz

#本构矩阵,D = np.zeros((6,6))
def eld(E,v):
    DM = np.zeros((6,6))
    coef=E*(1-v)/((1+v)*(1-2*v))
    a = v/(1-v)
    b = 0.5*(1-2*v)/(1-v)
    DM[0,0] = DM[1,1] = DM[2,2] = 1
    DM[0,1] = DM[0,2] = DM[1,0] = DM[1,2] = DM[2,0] = DM[2,1] = a
    DM[3,3] = DM[4,4] = DM[5,5] = b
    DM = coef * DM
    return DM

#获取B矩阵
def get_B(Nx,Ny,Nz):
    B = np.zeros((6,24))
    for i in range(8):
        B[0,3*i] = B[3,3*i+1] = B[5,3*i+2] = Nx[i]
        B[1,3*i+1] = B[3,3*i] = B[4,3*i+2] = Ny[i]
        B[2,3*i+2] = B[4,3*i+1] = B[5,3*i] = Nz[i]
    return B

#获取N矩阵(质量矩阵)
def get_NTN(N):
    NTN = np.zeros((3,24))
    for i in range(8):
        NTN[0][3*i] = N[i]
        NTN[1][3*i+1] = N[i]
        NTN[2][3*i+2] = N[i]
    return NTN

"""获取单元刚度矩阵,n表示高斯积分权数"""
def get_Ke(xo,yo,zo,D,n):
    #高斯积分点及权数
    P = np.zeros(3)
    H = np.zeros(3)
    if n == 3:
        P[0] = -0.7745966692414834
        P[1] = 0
        P[2] = -P[0]
        H[0] = H[2] = 5/9
        H[1] = 8/9
    else:
        P[0] = -0.5773502691896257
        P[1] = -P[0]
        H[0] = H[1] = 1.0
        
    Ke = np.zeros((24,24))
    B = np.zeros((3,24))
    Nx = np.zeros(8)
    Ny = np.zeros(8)
    Nz = np.zeros(8)
    for i in range(n):
        for j in range(n):
            for k in range(n):
                det_J,Nx,Ny,Nz= jaco(P[i],P[j],P[k],xo,yo,zo)
                B = get_B(Nx,Ny,Nz)
                Ke += H[i]*H[j]*H[k] * np.dot(B.T,np.dot(D,B)) * det_J
    return Ke

"""获取单元质量矩阵(无密度),n表示高斯积分权数"""
def get_Me(xo,yo,zo,D,n):
    #高斯积分点及权数
    P = np.zeros(3)
    H = np.zeros(3)
    if n == 3:
        P[0] = -0.7745966692414834
        P[1] = 0
        P[2] = -P[0]
        H[0] = H[2] = 5/9
        H[1] = 8/9
    else:
        P[0] = -0.5773502691896257
        P[1] = -P[0]
        H[0] = H[1] = 1.0
        
    Me = np.zeros((24,24))
    NTN = np.zeros((3,24))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                det_J,Nx,Ny,Nz= jaco(P[i],P[j],P[k],xo,yo,zo)
                N = np.zeros(8)
                FN(P[i],P[j],P[k],N)
                NTN = get_NTN(N)
                Me += H[i]*H[j]*H[k] * np.dot(NTN.T,NTN) * det_J
    #集中化处理
    Me = sum_rows(Me)
    return Me

"""集中化"""
def sum_rows(matrix):
    result = np.zeros_like(matrix)
    row_sums = np.sum(matrix, axis=1)
    np.fill_diagonal(result, row_sums)
    return result

#布尔矩阵A
def get_A(NN,Ele):
    row = np.array([])
    col = np.array([])
    for i in range(8):
        a = 3 * (Ele[i]-1)
        col = np.append(col,[a,a+1,a+2])
        row = np.append(row,[3*i,3*i+1,3*i+2])
    data = np.array([1 for i in range(24)])
    A = sp.sparse.csc_matrix((data,(row,col)),shape=(24,3*NN))
    return A

#单刚组装总刚
def Assemble_K(K,Ke,A):
    K += sp.sparse.csr_matrix(A.T*sp.sparse.csr_matrix(Ke*A))
    return K

#指定约束
def Fix(K,P,NN,k,n,a):
    j = 3*(k-1)+n
    P[j] = a
    '''
    for i in range(3*NN):
        if K[j,i] != 0:
            K[j,i] = 0
    '''
    K = K.tocsr()
    K.data[K.indptr[j]:K.indptr[j+1]] = 0
    K[j,j] = 1
    
    return K,P
    
#解稀疏矩阵方程
def Solve(K,P):
    return spsolve(K.tocsr(),P)

#计算应力
def CacuSS(E,v,NN,NE,X,Y,Z,Ele,V):
    stress_list = np.zeros((NN,6))
    time = np.zeros(NN)
    D = eld(E,v)
    a = [1,-1,-1,1,1,-1,-1,1]
    b = [-1,-1,1,1,-1,-1,1,1]
    c = [1,1,1,1,-1,-1,-1,-1]
    for i in range(NE):
        xo = [0]*8
        yo = [0]*8
        zo = [0]*8
        vl = [0]*24
        for j in range(8):
            n = int(Ele[i][j]-1)
            xo[j] = X[n]
            yo[j] = Y[n]
            zo[j] = Z[n]
            vl[3*j] = V[3*n]
            vl[3*j+1] = V[3*n+1]
            vl[3*j+2] = V[3*n+2]
        
        for j in range(8):
            n = int(Ele[i][j]-1)
            det_J,Nx,Ny,Nz = jaco(a[j],b[j],c[j],xo,yo,zo)
            B = get_B(Nx,Ny,Nz)
            ss = np.dot(D,np.dot(B,vl))
            for l in range(6):
                stress_list[n,l] += ss[l]
            time[n] += 1
    for i in range(NN):
        for j in range(6):
            stress_list[i,j] /= time[i]
    return stress_list



