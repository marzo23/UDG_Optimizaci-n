#REFERENCIAS USADAS A MODO DE MOSAICO PARA CONSTRUIR ESTE CÓDIGO:
# https://github.com/syahrulhamdani/Gradient-Descent-for-Rosenbrock-Function
# https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
#
import numpy as np
import random
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plot

def DerrivRosenbrock0 ( point ):
    dx = 2*point[0] - 400*point[0]*(point[1] - (point[0]**2))
    dy = 200*(point[1] - (point[0]**2))
    return dx, dy

def DerrivRosenbrock1 ( point ):
    dx = (-2*(1 - point[0]) - 400*(point[1] - (point[0]**2)**2))
    dy = 200*(point[1] - (point[0]**2))
    return dx, dy

def main():
    lrate = 0.002
    a = np.array([random.uniform(-1.0, 1.0), random.uniform(-1.0, 1.0)])
    epoch = 100000
    ai = []
    ai_listx = []
    ai_listy = []
    ai_list = []
    for i in range(epoch):
        f = ((1 - a[0])**2) + (100*((a[1] - a[0]**2)**2))
        ai.append([a,f])
        ai_listx.append(a[0])
        ai_listy.append(a[1])
        ai_list.append(f)
        fi = np.array(DerrivRosenbrock1(a))
        a = a - np.dot(lrate,fi)
    
    ai = np.array(ai)
    if not np.isnan(ai[-1, 1]):
        print(f'the minimum is: {ai[-1, 1]} at point: {ai[-1,0]}')

        fig = plot.figure()
        ax = fig.gca(projection='3d')

        s = 0.05
        X = np.arange(-2, 2.+s, s)
        Y = np.arange(-2, 3.+s, s)
            
        X, Y = np.meshgrid(X, Y)
        Z = ((1 - X)**2) + (100*((Y - X**2)**2))
        ax.scatter(ai[-1,0][0], ai[-1,0][0], ai[-1, 1], marker='o', s=100, c=['red'])
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False, alpha=.5)
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plot.show()

        return ai_list
    return None

def repeat():
    i = 0
    r_list = []
    f = open("tst2.csv", "w")
    while i < 30:
        r = main()
        if r is not None:
            i += 1
            f.write(f'{i},{r[-1]},{np.average(r)},{np.std(r)},{np.median(r)},{np.mean(r)}\n')
            print("EXECUTION "+str(i))
            print("MIN RESULT: "+str(r[-1]))
            print("Ex avg: "+str(np.average(r)))
            print("Ex std: "+str(np.std(r)))
            print("Ex median: "+str(np.median(r)))
            print("Ex mean: "+str(np.mean(r)))
            print("\n\n")
            r_list.append(r[-1])
    print("EXECUTION FINAL")
    print("Ex avg: "+str(np.average(r_list)))
    print("Ex std: "+str(np.std(r_list)))
    print("Ex median: "+str(np.median(r_list)))
    print("Ex mean: "+str(np.mean(r_list)))
    f.close()


if __name__ == '__main__':
    main()
