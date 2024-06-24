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


# busqueda_unidireccional
def primera_derivada(x, funcion):
    delta = 0.0001
    return (funcion(x + delta) - funcion(x - delta)) / (2 * delta)

def segunda_derivada(x, funcion):
    delta = 0.0001
    return (funcion(x + delta) - 2 * funcion(x) + funcion(x - delta))/(delta**2)

def newton_raphson(x, funcion, e):
    k = 0
    while True:
        x_derivada1 = primera_derivada(x, funcion)
        x_derivada2 = segunda_derivada(x, funcion)
        x_siguiente = x - (x_derivada1 / x_derivada2)
        if abs(x_siguiente - x) < e:
            return x_siguiente
        x = x_siguiente
        k += 1
        if k > 1000:
            return None



def busqueda_unidireccional(x, s, funcion, e):
    def objetivo(alpha):
        return funcion(x + alpha * s)
    alpha_optimo = newton_raphson(0, objetivo, e)
    return alpha_optimo

def optimizador_unidireccional(x_t, s_t, funcion, e):
    alpha_optimo = busqueda_unidireccional(x_t, s_t, funcion, e)
    x_alpha_optimo = x_t + alpha_optimo * s_t
    return x_alpha_optimo

def gradiente_conjugado(funcion,x,epsilon1,epsilon2,epsilon3):
    #step 1
    x0 = x
    #step 2
    grad = np.array(gradiente(funcion,x))
    s0 = -(grad)
    #step 3
    gama = optimizador_unidireccional(x0,s0,funcion,0.001)
    k=1
    sk_min1= s0
    xk = gama
    dev_xk=np.array(gradiente(funcion,xk))
    terminar = 0
    
    while terminar != 1:
        #step 4
        sk = -(dev_xk) + np.dot(np.divide(np.sum(dev_xk)**2,np.sum(grad)**2), sk_min1)
        sk_min1 = sk
        #step5
        gama_xk=optimizador_unidireccional(xk,sk,funcion,0.001)
        # print(gama_xk)
        #step 6
        q1 = (gama_xk[1] - gama_xk[0])/gama_xk[0]
        q2 = np.mean(gama_xk)
        xk = gama_xk
        if (q1/q2) < epsilon2:
            return sk
        else:
            terminar = 0
            k = k+1
        if k == 1000:
            terminar = 1



himmenblau = lambda x: (((x[0]**2)+x[1]-11)**2) + ((x[0]+(x[1]**2)-7)**2)
print(gradiente_conjugado(himmenblau,np.array([0.0,0.0]),0.001,0.001,0.001))