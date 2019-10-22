import matplotlib.pyplot as plt
import autograd as autograd
import autograd.numpy as np


g= lambda X : (X[1])**2 +(X[0])**2
g1=lambda X : np.exp(-X[0]**2-X[1]**2)
g2=lambda X : np.exp( -(X[0]-1)**2-(X[1]-1)**2)

def f(X):
    return 2*(np.exp(-X[0]**2-X[1]**2)-np.exp( -(X[0]-1)**2-(X[1]-1)**2))


def find_seed_D (g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if (g([limx[1],limy[0]])-c)*(g([limx[0],limy[0]])-c)<=0:
        a,b=limx[0],limx[1]
        if g([a,limy[0]])>g([b,limy[0]]):
            a,b=b,a
        while abs(b-a)>eps:
            d=(a+b)/2
            if (g([d,limy[0]])-c)>0:
                b=d
            else:
                a=d
        return [d,limy[0]]
    else :
        return None
def find_seed_L (g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if (g([limx[0],limy[0]])-c)*(g([limx[0],limy[1]])-c)<=0 :
        a,b=limy[0],limy[1]
        if g([limx[0],a])>g([limx[0],b]):
            a,b=b,a
        while abs(b-a)>eps:
            d=(a+b)/2
            if (g([limx[0],d])-c)>0:
                b=d
            else:
                a=d
        return [limx[0],d]
    else :
        return None
def find_seed_U (g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if (g([limx[0],limy[1]])-c)*(g([limx[1],limy[1]])-c)<=0 :
        a,b=limx[0],limx[1]
        if g([a,limy[1]])>g([b,limy[1]]):
            a,b=b,a
        while abs(b-a)>eps:
            d=(a+b)/2
            if (g([d,limy[1]])-c)>0:
                b=d
            else:
                a=d
        return [d,limy[1]]
    else:
        return None
def find_seed_R(g,c=0,limx=[0.,1.],limy=[0.,1.],eps=2**(-26)):
    if  (g([limx[1],limy[1]])-c)*(g([limx[1],limy[0]])-c)<=0 :
        a,b=limy[0],limy[1]
        if g([limx[1],a])>g([limx[1],b]):
            a,b=b,a
        while abs(b-a)>eps:
            d=(a+b)/2
            if (g([limx[1],d])-c)>0:
                b=d
            else:
                a=d
        return [limx[1],d]
    else:
        return None
def cart_pol(position,delta):
    return np.arccos(position[0]/delta)
def pol_cart(theta,delta):
    return np.array([delta*np.cos(theta),delta*np.sin(theta)])
def simple_contour(g,xc,yc,nom_bord,i,j,c=0,delta=0.01,eps=2**(-26)):
    abscisses , ordonnées = [],[]
    dic_fonction = {"UP" : find_seed_U, "LEFT":find_seed_L, "RIGHT" : find_seed_R,"DOWN" : find_seed_D }
    position = dic_fonction[nom_bord](g,c,[xc[i],xc[i+1]], [yc[j],yc[j+1]],2**(-26))
    gradg=autograd.grad(g)
    
    
    def h(theta,position):
                return (f(position+[delta*np.cos(theta),delta*np.sin(theta)])-c)
    def derh(theta,position1):
        grad = gradg(position1+np.array([delta*np.cos(theta),delta*np.sin(theta)]))
        deriv=np.array([(-1)*delta*np.sin(theta),delta*np.cos(theta)])
        return grad@deriv
    
    if  not isinstance(position,list):
        return [],[]
    else:
        position = np.array(position) 
        print("amorce",position) 
        abscisses.append(position[0])
        ordonnées.append(position[1])
        def test (position):
            return xc[i]<=position[0]<=xc[i+1] and yc[j]<=position[1]<=yc[j+1]

        while test(position) :
            gradX=gradg(position)
            norme = np.sqrt(gradX[1]**2+gradX[0]**2)
            vect = np.array([gradX[1]/norme,-1*gradX[0]/norme])
            
            
            position_faux = position + vect*delta
            print(position_faux)
            #Méthode de Newton  
            position_relat=position_faux-position
            theta0 = cart_pol(position_relat,delta)
            theta=theta0-h(theta0,position)/derh(theta0,position)
            
            while (theta-theta0)>=eps:
                theta0 = theta
                theta = theta0-h(theta0,position)/derh(theta0,position)
            position = position+pol_cart(theta,delta)
            print(position)
            abscisses.append(position[0])
            ordonnées.append(position[1])
            print(test(position))
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


