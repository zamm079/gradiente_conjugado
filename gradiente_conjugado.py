import numpy as np
import math

def regla_eliminacion(x1,x2,fx1,fx2,a,b) -> tuple[float,float]:
    if fx1 > fx2:
        return x1, b
    
    if fx1 < fx2:
        return a, x2
    
    return x1, x2

def w_to_x(w:float,a,b) -> float:
    return w * (b-a) + a

def busquedaDorada(funcion, epsilon:float, a:float=None, b:float=None) -> float:
    PHI = (1 + math.sqrt(5)) / 2 - 1
    aw, bw = 0, 1
    Lw = 1
    k = 1

    while Lw > epsilon:
        w2 = aw + PHI*Lw
        w1 = bw + PHI*Lw
        aw, bw = regla_eliminacion(w1, w2, funcion(w_to_x(w1,a,b)),
                                   funcion(w_to_x(w2,a,b)),aw,bw)
        k+=1
        Lw = bw - aw
    return (w_to_x(aw,a,b)+w_to_x(bw,a,b))/2

def gradiente(f,x,deltaX=0.001):
    grad=[]
    for i in range(0,len(x)):
        xp = x.copy()
        xn = x.copy()
        xp[i] = xp[i]+deltaX
        xn[i] = xn[i]-deltaX
        grad.append((f(xp)-f(xn))/(2*deltaX))
    return grad

def gradiente_conjugado(funcion,x,epsilon1,epsilon2,epsilon3):
    #step 1
    grad = np.array(gradiente(funcion,x))
    #step2
    S0 = -(grad)
    

    return



himmenblau = lambda x: (((x[0]**2)+x[1]-11)**2) + ((x[0]+(x[1]**2)-7)**2)
X0 = np.array([0.0,0.0])
gradiente_conjugado(himmenblau,X0,0.001,0.001,0.001)