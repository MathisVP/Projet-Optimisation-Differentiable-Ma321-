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
from matplotlib import pyplot as plt
import time


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


def Laplacienne(n):
    """Contruction d'une matrice laplacienne"""
    
    L = np.eye(n,n)
    L = 2*L

    for i in range(n-1):
        L[i][i+1] = -1
        L[i+1][i] = -1
                
    return L
    

def b(n):
    """Construction de la matrice b"""
    b = np.zeros((n, 1))
    
    for i in range(n): #Construction de la matrice b
        b[i][0] = 1
    
    return b


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


def ResolTriangInf(A,B):
    """Fonction qui résout une système triangulaire inférieur"""

    B = np.asarray(B,dtype=float).reshape(5,1)
    Taug = np.concatenate((A,B), axis=1)
    n = Taug.shape[0]
    Y = np.zeros(n)

    for i in range(n):
        Y[i] = Taug[i,n]

        for j in range(0,i):
            Y[i] = Y[i] - Taug[i,j] * Y[j] 
        Y[i]= Y[i]/Taug[i,i]

    Y = np.asarray(Y,dtype=float).reshape(n,1)

    return Y

def ResolTriangSup(A,B):
    """Fonction qui résout une système triangulaire supérieur"""

    Taug = np.concatenate((A,B), axis = 1)
    n = Taug.shape[0]
    x = np.zeros(n)
    x[n-1] = Taug[n-1,n]/ Taug[n-1,n-1]

    for i in range(n-2, -1, -1):
        x[i] = Taug[i,n]

        for j in range(i+1,n):
            x[i] = x[i] - Taug[i,j] * x[j] 

        x[i]= x[i]/Taug[i,i]

    x = np.asarray(x,dtype=float).reshape(n,1)

    return x


def ResolCholesky(A, B):
    """Fonction qui résout Cholesky"""

    L = Cholesky(A)
    Lt = np.transpose(L)
    Y = ResolTriangInf(L,B)
    X = ResolTriangSup(Lt,Y)

    return X 


def main():
    """Fonction main"""
    
    #Création du tableau comparaison (taille, nb d'itération, temps de calcul) présent sur le site du CNRS 
    iterations = []
    temps = []
    for n in range(100, 1001, 100):
        A = Laplacienne(n)
        x0 = np.zeros((n, 1))
        debut = time.time()
        x, k = Gradient_Conjugue(A, b(n), x0, 1e-10)
        fin = time.time()
        iterations.append(k)
        temps.append(fin-debut)
        
    print("Iterations :",iterations)
    print("Temps de calcul:", temps)

    #Construction du graphique du conditionnement en fonction de la taille de la matrice avec une courbe "réel" et "théorique"
    cond = []
    taille = []
    for n in range(2,1001, 2):
        taille.append(n)
        c = np.linalg.cond(Laplacienne(n))
        cond.append(c)

    B = (np.log(cond[-1])-np.log(cond[4]))/(np.log(taille[-1])-np.log(taille[4]))
    a = np.exp(np.log(cond[-1])- B*np.log(taille[-1]))
    print("b =", B)
    print("a =", a)

        
    plt.plot(taille, cond, label="Réel")
    x = range(2, 1001, 2)
    y = []

    for i in x:
        y.append(a*(i**B))
    
    plt.plot(x, y, label= r"Modèle $y = x^b$")
    plt.title(r"Conditionnement d'une matrice Laplacienne en fonction de sa taille")
    plt.ylabel(r"cond")
    plt.xlabel(r"n")
    plt.legend()
    plt.grid()
    plt.show()

    # #Coparaison entre la méthode du gradient conjugée et la méthode de Cholesky
    # cond_Ch = []
    # taille_Ch = []
    # for n in range(2,201, 2):
    #     taille_Ch.append(n)
    #     c = np.linalg.cond(ResolCholesky(Laplacienne(n), b(n)))
    #     cond_Ch.append(c)

    # plt.plot(taille_Ch, cond_Ch, label= r"")
    # plt.title(r"Conditionnement d'une matrice Laplacienne en fonction de sa taille avec la méthode de Cholesky")
    # plt.ylabel(r"cond")
    # plt.xlabel(r"n")
    # plt.legend()
    # plt.grid()
    # plt.show()


if __name__ == '__main__':
    main()
