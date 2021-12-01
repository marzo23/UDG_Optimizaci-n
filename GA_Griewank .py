
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
import math
import pandas as pd

def griewank_func(x, n = 10):
    fr = 4000
    s = 0
    p = 1
    for j in range(len(x)):
        s = s+x[j]**2
    for j in range(len(x)):
        p = p*math.cos(x[j]/math.sqrt(j+1))
    return s/fr-p+1

def onemax(a):
	return griewank_func(a)

def selection(pop, scores, k=3):
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def crossover(p1, p2):
    mid = len(p2/2)
    p1n = []
    p2n = []
    for i in range(len(p2)):
        if i < mid:
            p1n.append(p1)
            p2n.append(p2)
        else:
            p1n.append(p2)
            p2n.append(p1)
    return p1,p2

def mutation(a, r_mut):
	for i in range(len(a)):
		if rand() < r_mut:
			a[i] = random.uniform(-2.0, 2.0)

def genetic_algorithm(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	pop = np.random.uniform(-600,600, (n_pop, n_bits))
	best, best_eval = 0, objective(pop[0])
	for gen in range(n_iter):
		scores = [objective(c) for c in pop]
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
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
n_bits = 10
n_pop = 10
r_cross = 0.9
r_mut = 1.0 / float(n_bits)

ITERATIONS = 100
t2 = []
for k in range(ITERATIONS):
    a, b = best, score = genetic_algorithm(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
    tmp = [b]+ [i for i in a]
    t2.append(tmp)

t2_df = pd.DataFrame(data=t2)
t2_df.describe().to_csv("C:\\Users\\cristinam\\Downloads\\ga_iter_desc3.csv")