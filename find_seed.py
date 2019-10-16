import autograd as autograd 
from autograd import numpy as np 
import matplotlib.pyplot as plt 

###Méthode de Newton à 1D avec condition de non trouvage de Seed, dans le cadre d'un carré de taille [0,1]²
#Condition de recherche : on veut (f(0,0)-c)(f(0,1)-c)<0 (TVI)

g= lambda X : (X[1])**2 +(X[0])**2

def find_seed(g,c=0,eps=2**(-26)):
    if (g([0,1])-c)*(g([0,0])-c)<0:
        return None
    else : 
        gradg=autograd.grad(g)
        x_0=5
        x=x_0-(g(np.array([0.,x_0]))-c)/gradg(np.array([0.,x_0]))[1]
        while abs(x-x_0)>eps:
            x_0=x
            x=x_0-(g(np.array([0,x_0]))-c)/gradg(np.array([0,x_0]))[1]
        return [0,x] 