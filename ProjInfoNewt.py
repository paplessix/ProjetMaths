import matplotlib.pyplot as plt
import autograd as autograd
from autograd import numpy as np


g= lambda X : (X[1])**2 -(X[0])**2
g1=lambda X : np.exp(-X[0]**2-X[1]**2)
g2=lambda X : np.exp( -(X[0]-1)**2-(X[1]-1)**2)

def f(X):
    return 2*(np.exp(-X[0]**2-X[1]**2)-np.exp( -(X[0]-1)**2-(X[1]-1)**2))
def find_seed_L (g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if (g([limx[0],limy[0]])-c)*(g([limx[0],limy[1]])-c)<0:
        gradg=autograd.grad(g)
        y_0=limy[1]
        if gradg(np.array([limx[0],y_0]))[1]==0:
            y_0=(limy[1]+limy[0])/2
        y=y_0-(g(np.array([limx[0],y_0]))-c)/gradg(np.array([limx[0],y_0]))[1]
        while abs(y-y_0)>eps:
            y_0=y
            y=y_0-(g(np.array([limx[0],y_0]))-c)/gradg(np.array([limx[0],y_0]))[1]
        return [limx[0],y]
    else :
        return None
def find_seed_U (g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if (g([limx[0],limy[1]])-c)*(g([limx[1],limy[1]])-c)<=0 :
        gradg = autograd.grad(g)
        x_0 = limx[1]
        if gradg(np.array([x_0,limy[1]]))[0]==0:
            x_0=(limx[1]+limx[0])/2
        x = x_0-(g(np.array([x_0,limy[1]]))-c)/gradg(np.array([x_0,limy[1]]))[0]
        while abs(x-x_0)>eps:
            x_0 = x
            x = x_0-(g(np.array([x_0,limy[1]]))-c)/gradg(np.array([x_0,limy[1]]))[0]
        return ([x,limy[1]])
    else :
        return None
def find_seed_R (g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if (g([limx[1],limy[1]])-c)*(g([limx[1],limy[0]])-c)<=0 :
        gradg=autograd.grad(g)
        y_0=limy[1]
        if gradg(np.array([limx[1],y_0]))[1]==0:
            y_0=(limy[1]+limy[0])/2
        y=y_0-(g(np.array([limx[1],y_0]))-c)/gradg(np.array([limx[1],y_0]))[1]
        while abs(y-y_0)>eps:
            y_0=y
            y=y_0-(g(np.array([limx[1],y_0]))-c)/gradg(np.array([limx[1],y_0]))[1]
        return ([limx[1],y])
    else:
        return None
def find_seed_D (g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if  (g([limx[1],limy[0]])-c)*(g([limx[0],limy[0]])-c)<=0 :
        gradg = autograd.grad(g)
        x_0 = limx[1]
        if gradg(np.array([x_0,limy[0]]))[0]==0:
            x_0=(limx[1]+limx[0])/2
        x = x_0-(g(np.array([x_0,limy[0]]))-c)/gradg(np.array([x_0,limy[0]]))[0]
        while abs(x-x_0)>eps:
            x_0 = x
            x = x_0-(g(np.array([x_0,limy[0]]))-c)/gradg(np.array([x_0,limy[0]]))[0]
        return ([x,limy[0]])
    else:
        return None

def simple_contour(g,xc,yc,nom_bord,i,j,c=0,delta=0.01):
    abscisses , ordonnées = [],[]
    dic_fonction = {"UP" : find_seed_U, "LEFT":find_seed_L, "RIGHT" : find_seed_R,"DOWN" : find_seed_D }
    position = dic_fonction[nom_bord](g,c,[xc[i],xc[i+1]], [yc[j],yc[j+1]])
    gradg=autograd.grad(g)
    if  not isinstance( position,list) :
        return [],[]
    else:
        position = np.array(position)        
        abscisses.append(position[0])
        ordonnées.append(position[1])

        def test (position):
            return xc[i]<=position[0]<=xc[i+1] and yc[j]<=position[1]<=yc[j+1]
        while test(position) :
            gradX=gradg(position)
            norme = np.sqrt(gradX[1]**2+gradX[0]**2)
            vect = np.array([gradX[1]/norme,-1*gradX[0]/norme])
            position = position + vect*delta

            abscisses.append(position[0])
            ordonnées.append(position[1])
        return abscisses,ordonnées

def contour(f, c=0, xc = [0,1], yc = [0,1], delta = 0.01):
    xs,ys= [], []

    liste_bord = ['UP','LEFT','RIGHT','DOWN']
    for i in range(len(xc)-1):
        for j in range(len(yc)-1):
            for nom_bord in liste_bord :
                X,Y=simple_contour(f,xc,yc,nom_bord,i,j,c,delta)
                xs.append(X)
                ys.append(Y)
    return xs,ys

def trace(g,xc,yc,c=0):
    a,b=contour(g,c,xc,yc)
    for x,y in zip(a,b):
        plt.plot(x,y)
    plt.show()