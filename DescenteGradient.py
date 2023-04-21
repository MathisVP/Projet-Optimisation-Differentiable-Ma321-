import numpy as np
from matplotlib import pyplot as plt

def steepest_descent_constant_step(A, b, x0, epsilon, h):
    """Méthode de descente à pas fixe"""

    iteration = 0
    x = x0
    while np.linalg.norm(A@x-b) > epsilon:
        x = x - h*(A@x-b)
        iteration += 1
    return x, iteration


def steepest_descent_optimal_step(A, b, x0, epsilon):
    """Méthode de descente à pas optimal"""

    iteration = 0
    x = x0
    while np.linalg.norm(A@x-b) > epsilon:
        r = A@x-b
        h = (r.T@r)/(r.T@A@r)
        x = x - h*(A@x-b)
        iteration += 1

    return x, iteration

def steepest_descent_constant_step2(A, b, x0, xs, epsilon, h):
    """Méthode de descente à pas fixe avec un autre critère d'arret"""

    iteration = 0
    x = x0
    xold = 0
    xlist = []
    x1list = []

    while np.linalg.norm(x-xold) > epsilon:
        xold = x
        x = xold - h*(A@x-b)
        xlist.append(np.linalg.norm(xold-xs))
        x1list.append(np.linalg.norm(x-xs))
        iteration += 1

    return x, iteration, xlist, x1list

def iter_cond(cmin, cmax, p):
    """Calcul du nombre d'itération en fonction du conditionnement de A"""
    
    i = []
    c = []

    for l in range(10, 1001, 10):
        A = np.array([[1, 0], [0, l]])
        b = np.array([1, l])
        x0 = np.array([3.2, 1.4])
        epsilon = 1e-6
        h = 2/(1+l)
        x, iteration = (steepest_descent_constant_step(A, b, x0, epsilon, h))
        i.append(iteration)
        c.append(l)

    plt.plot(c, i)
    plt.xlabel("Conditionnement")
    plt.ylabel("Nombre d'itération")
    plt.show()

def convergence(A, b, x0, xs, epsilon, h):
    """Calcul de l'ordre p"""
    
    x, iteration, xlist, x1list = steepest_descent_constant_step2(A, b, x0, np.array([1, 1]), epsilon, h)
    plt.loglog(xlist, x1list)
    plt.grid()
    plt.xlabel(r"||$x_k - x*||$")
    plt.ylabel(r"||$x_{k+1} - x*$||")
    plt.show()

    slope, intersept = np.polyfit(xlist, x1list, 1) #Déterminer la pente de la droite
    print("pente =",slope)
    #déterminer C

def main():
    """Fonction main"""

    iter_cond(10, 1001, 10)

    A = np.array([[2, 0], [0, 4]])
    b = np.array([2, 4])
    x0 = np.array([3.2, 1.4])
    epsilon = 1e-6
    h = 0.33
    xs = np.array([1, 1])

    convergence(A, b, x0, xs, epsilon, h)

if __name__ == '__main__':
    main()