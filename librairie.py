"""
Projet
Ma321
Lucas Jeannot
Hugo Tassin
Mathis Veyrat-Parisien
Alexis Wahlers
"""

#------------------------------------------
#Importation des bibliotheques
#------------------------------------------

import numpy as np

#------------------------------------------
#Méthode de Cholesky
#------------------------------------------

def ResolTriangInf(A, B):
    """Fonction qui résout une système triangulaire inférieur"""

    B = np.asarray(B, dtype=float).reshape(5, 1)
    Taug = np.concatenate((A, B), axis=1)
    n = Taug.shape[0]
    Y = np.zeros(n)

    for i in range(n):
        Y[i] = Taug[i, n]

        for j in range(0, i):
            Y[i] = Y[i] - Taug[i, j] * Y[j] 
        Y[i]= Y[i]/Taug[i, i]

    Y = np.asarray(Y, dtype=float).reshape(n, 1)

    return Y


def ResolTriangSup(A, B):
    """Fonction qui résout une système triangulaire supérieur"""

    Taug = np.concatenate((A, B), axis = 1)
    n = Taug.shape[0]
    x = np.zeros(n)
    x[n-1] = Taug[n-1, n]/ Taug[n-1, n-1]

    for i in range(n-2, -1, -1):
        x[i] = Taug[i,n]

        for j in range(i+1, n):
            x[i] = x[i] - Taug[i, j] * x[j] 

        x[i]= x[i]/Taug[i, i]

    x = np.asarray(x, dtype=float).reshape(n,1)

    return x


def ResolCholesky(A, B):
    """Fonction qui résout Cholesky"""

    L = Cholesky(A)
    Lt = np.transpose(L)
    Y = ResolTriangInf(L, B)
    X = ResolTriangSup(Lt, Y)

    return X 


def Cholesky(A):
    """Méthode de Cholesky"""

    n = A.shape[0]
    L = np.zeros((n, n), dtype=float)

    for j in range(n):
        s = 0

        for k in range(j):
            s = s + L[j, k] ** 2
        L[j, j] = np.sqrt(A[j, j] - s)

        for i in range(j + 1, n):
            s = 0

            for k in range(1, j):
                s = s + L[i, k] * L[j, k]

            L[i, j] = (A[i, j] - s) / L[j, j]

    return L

#------------------------------------------
#Méthode Gradient pas optimal
#------------------------------------------



#------------------------------------------
#Méthode Gradient pas conjugé
#------------------------------------------

def Gradient_Conjugue(A, b, x0, epsilon):
    """Méthode du grandient conjugué"""

    r = b - A@x0
    p = r
    x = x0
    k = 0
    
    while np.linalg.norm(r)/np.linalg.norm(b) > epsilon:
        rold = r
        alpha = (rold.T@rold)/(p.T@A@p)
        x = x + alpha*p
        r = rold - alpha*A@p
        beta = (r.T@r)/(rold.T@rold)
        p = r + beta*p
        k += 1
        
    return x, k


#------------------------------------------
#Fonction pour définir une matrice laplacienne
#------------------------------------------

def Laplacienne(n):
    """Contruction d'une matrice laplacienne"""
    
    L = np.eye(n,n)
    L = 2*L

    for i in range(n-1):
        L[i][i+1] = -1
        L[i+1][i] = -1
                
    return L


#------------------------------------------
#Fonction pour définir une matrice b
#------------------------------------------

def b(n):
    """Construction de la matrice b"""
    b = np.zeros((n, 1))
    
    for i in range(n): #Construction de la matrice b
        b[i][0] = 1
    
    return b
