#REFERENCIAS USADAS A MODO DE MOSAICO PARA CONSTRUIR ESTE CÓDIGO:
#https://machinelearningmastery.com/simple-genetic-algorithm-from-scratch-in-python/
# https://github.com/syahrulhamdani/Gradient-Descent-for-Rosenbrock-Function
# https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
#
from numpy.random import randint
from numpy.random import rand
import random
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from matplotlib import cm

def onemax(a):
	res = ((1 - a[0])**2) + (100*((a[1] - a[0]**2)**2))
	#print("["+str(a[0])+","+str(a[1])+"]="+str(res))
	return res

def selection(pop, scores, k=3):
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def crossover(p1, p2):
	return [[p1[0], p2[1]],[p2[0], p1[1]]]

def mutation(a, r_mut):
	for i in range(len(a)):
		if rand() < r_mut:
			a[i] = random.uniform(-2.0, 2.0)

def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	pop = [[random.uniform(-2.0, 2.0), random.uniform(-2.0, 2.0)] for _ in range(n_pop)]
	best, best_eval = 0, objective(pop[0])
	for gen in range(n_iter):
		scores = [objective(c) for c in pop]
		fig = plot.figure()
		ax = fig.gca(projection='3d')
		ax.view_init(30,50)
		for i in range(n_pop):
			if i % 10 == 0:
				ax.scatter(pop[i][0], pop[i][1], scores[i], marker='o', s=50, c=['blue'])
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
		s = 0.05
		X = np.arange(-2, 2.+s, s)
		Y = np.arange(-2, 3.+s, s)
		X, Y = np.meshgrid(X, Y)
		Z = ((1 - X)**2) + (100*((Y - X**2)**2))
		ax.scatter(best[0], best[1], best_eval, marker='o', s=100, c=['red'])
		surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
				linewidth=0, antialiased=False, alpha=.5)
		ax.zaxis.set_major_locator(LinearLocator(10))
		ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
		fig.colorbar(surf, shrink=0.5, aspect=5)
		plot.savefig("C:\\Users\\crist\\Documents\\GENETIC PIC\\gen_"+str(gen)+".png")
		selected = [selection(pop, scores) for _ in range(n_pop)]
		children = list()
		for i in range(0, n_pop, 2):
			p1, p2 = selected[i], selected[i+1]
			for c in crossover(p1, p2):
				mutation(c, r_mut)
				children.append(c)
		pop = children
	return [best, best_eval]

n_iter = 100
n_bits = 20
n_pop = 100
r_cross = 0.9
r_mut = 1.0 / float(n_bits)
best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done!')
print('f(%s) = %f' % (best, score))
