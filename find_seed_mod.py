"""
La foncrtion prend en argument une fonction g, une valeur c qui définit l'équation g(x,y)=c, 
les entrées limx et limy sont des listes de 2 éléments chacunes 
qui  correspondent aux limites de la boite, epsilon corresond à la précision souhaitée
On teste d'abord find seed sur l'arrete gauche puis la supérieure, puis celle de droite 
et enfin l'arrete inférieure . on renvoit none si on ne trouve pas d'amorce
"""

import autograd
from autograd import numpy as anp 
import matplotlib.pyplot as plt 


def find_seed_mod(g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if (g([limx[0],limy[0]])-c)*(g([limx[0],limy[1]])-c)<0:
        gradg=autograd.grad(g)
        y_0=limy[1]
        y=y_0-(g(np.array([limx[0],y_0]))-c)/gradg(np.array([limx[0],y_0]))[1]
        while abs(y-y_0)>eps:
            y_0=y
            y=y_0-(g(np.array([limx[0],y_0]))-c)/gradg(np.array([limx[0],y_0]))[1]
        return [limx[0],y]


    elif (g([limx[0],limy[1]])-c)*(g([limx[1],limy[1]])-c)<0 :
        gradg = autograd.grad(g)
        x_0 = limx[0]
        x = x_0-(g(np.array([x_0,limy[1]]))-c)/gradg(np.array([x_0,limy[1]]))[0]
        while abs(x-x_0)>eps:
            x_0 = x
            x = x_0-(g(np.array([x_0,limy[1]]))-c)/gradg(np.array([x_0,limy[1]]))[0]
        return ([x,limy[1]])

    elif (g([limx[1],limy[1]])-c)*(g([limx[1],limy[0]])-c)<0 :
        gradg=autograd.grad(g)
        y_0=limy[1]
        y=y_0-(g(np.array([limx[1],y_0]))-c)/gradg(np.array([limx[1],y_0]))[1]
        while abs(y-y_0)>eps:
            y_0=y
            y=y_0-(g(np.array([limx[1],y_0]))-c)/gradg(np.array([limx[1],y_0]))[1]
        return ([limx[1],y])

    elif  (g([limx[1],limy[0]])-c)*(g([limx[0],limy[0]])-c)<0 :
        gradg = autograd.grad(g)
        x_0 = limx[0]
        x = x_0-(g(np.array([x_0,limy[0]]))-c)/gradg(np.array([x_0,limy[0]]))[0]
        while abs(x-x_0)>eps:
            x_0 = x
            x = x_0-(g(np.array([x_0,limy[0]]))-c)/gradg(np.array([x_0,limy[0]]))[0]
        return ([x,limy[0]])

    else:
        return None

