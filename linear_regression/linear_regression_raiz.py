# Programa de regressão linear simples
import matplotlib.pyplot as plt
import numpy as np

def plot(x,y,pair):
    # Pontos
    plt.plot(x,y,'o',color='#ea004f')
    plt.xlabel('x')
    plt.ylabel('y')DO
    maxx = max(x)+1
    maxy = max(y)+1
    minx = min(x)-1
    miny = min(y)-1
    plt.axis([minx,maxx,miny,maxy])
    # Regressão
    i = np.arange(0,11)
    y = pair[0]*i+pair[1]
    if(pair[1]<0):
         l = 'y='+str(round(pair[0],4))+'x'+str(round(pair[1],4))
    else:
        l = 'y='+str(round(pair[0],4))+'x'+'+'+ str(round(pair[1],4))
    
    plt.plot(i, y, label=l, color='#000')
    plt.legend()
    plt.show()

def find_coef(x,y):
    n = np.size(x)

    mx,my = np.mean(x), np.mean(y)

    ss_xy = np.sum(x*y) - n*mx*my
    ss_xx = np.sum(x*x) - n*mx*mx

    m = ss_xy/ss_xx
    b = my - m*mx

    return (m,b)



x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12])

pair = find_coef(x,y) 
print("Resultado da sua regressão: {} {}".format(pair[0], pair[1]));
# print(f"Resultado da sua regressão: {pair[0]} {pair[1]}");
plot(x,y,pair);








