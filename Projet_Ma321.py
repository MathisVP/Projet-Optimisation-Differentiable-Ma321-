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
from librairie import *


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
