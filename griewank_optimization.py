#http://www-optima.amp.i.kyoto-u.ac.jp/member/student/hedar/Hedar_files/TestGO_files/TestGO_files/TestCodes/griew.m

#############GRADIENT##################

import numpy as np
import random
import math

def griewank_func(x, n = 20):
    fr = 4000
    s = 0
    p = 1
    for j in range(n):
        s = s+x[j]**2
    for j in range(n):
        p = p*math.cos(x[j]/math.sqrt(j+1))
    return s/fr-p+1

def derv_griewank_func_arr(x, n = 20):
    return [derv_griewank_func(x, i, n) for i in range(n)]


def derv_griewank_func(x, i, n = 20):
    fr = 4000
    p = 1
    s = 2*x[i]
    for j in range(n):
        if j == 1:
            p = p*-math.sin(x[j]/math.sqrt(j+1))
        else:
            p = p*math.cos(x[j]/math.sqrt(j+1))
    return s/fr-p

def DerrivRosenbrock0 ( point ):
    dx = 2*point[0] - 400*point[0]*(point[1] - (point[0]**2))
    dy = 200*(point[1] - (point[0]**2))
    return dx, dy

# z = (1-x)**2 + 100*((y-x**2)**2)
# z = 1 - 2x + x**2 + 100y**2 - 200yx**2 + 100x**4)
# dx = -2 + 2x -400x + 400x**3

def DerrivRosenbrock1 ( point ):
    dx = (-2*(1 - point[0]) - 400*(point[1] - (point[0]**2)**2))
    dy = 200*(point[1] - (point[0]**2))
    return dx, dy

def main():
    lrate = 0.002
    a = np.random.uniform(-600,600, (1, 20))[0]
    epoch = 10000
    ai = []
    ai_listx = []
    ai_listy = []
    ai_list = []
    for i in range(epoch):
        f = griewank_func(a)
        ai.append([a,f])
        ai_listx.append(a[0])
        ai_listy.append(a[1])
        ai_list.append(f)
        fi = np.array(derv_griewank_func_arr(a, len(a)))
        a = a - np.dot(lrate,fi)
    
    ai = np.array(ai)
    if not np.isnan(ai[-1, 1]):
        print(f'the minimum is: {ai[-1, 1]} at point: {ai[-1,0]}')

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
    
    
#################GENETIC##################

from numpy.random import randint
from numpy.random import rand
import random
import numpy as np
import math

def griewank_func(x, n = 20):
    fr = 4000
    s = 0
    p = 1
    for j in range(n):
        s = s+x[j]**2
    for j in range(n):
        p = p*math.cos(x[j]/math.sqrt(j+1))
    return s/fr-p+1

def onemax(a):
	res = griewank_func(a) # ((1 - a[0])**2) + (100*((a[1] - a[0]**2)**2))
	#print("["+str(a[0])+","+str(a[1])+"]="+str(res))
	return res

def selection(pop, scores, k=3):
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]

def crossover(p1, p2, r_cross):
	# children are copies of parents by default
	c1, c2 = p1.copy(), p2.copy()
	# check for recombination
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = list(p1[:pt])
		c1.extend(p2[pt:])
		#c1 = p1[:pt] + p2[pt:]
		c2 = list(p2[:pt])
		c2.extend(p1[pt:])
		#c2 = p2[:pt] + p1[pt:]
	return [c1, c2]

def mutation(a, r_mut):
	for i in range(len(a)):
		if rand() < r_mut:
			a[i] = random.uniform(-2.0, 2.0)

def genetic_algorithm(objective, n_features, n_iter, n_pop, r_cross, r_mut):
	pop = np.random.uniform(-600,600, (n_pop, n_features))
	best, best_eval = 0, objective(pop[0])
	for gen in range(n_iter):
		scores = [objective(c) for c in pop]
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				#print(">%d, new best f(%s) = %.3f" % (gen, pop[i], scores[i]))
		selected = [selection(pop, scores) for _ in range(n_pop)]
		children = list()
		for i in range(0, n_pop, 2):
			p1, p2 = selected[i], selected[i+1]
			for c in crossover(p1, p2, r_cross):
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
print('Done! ', score)
#print('f(%s) = %f' % (best, score))


#############PSO########################

import sys
import numpy as np
import math

class PSO:

    def __init__(self, particles, velocities, fitness_function,
                 w=0.8, c_1=1, c_2=1, max_iter=100, auto_coef=True):
        self.particles = particles
        self.velocities = velocities
        self.fitness_function = fitness_function

        self.N = len(self.particles)
        self.w = w
        self.c_1 = c_1
        self.c_2 = c_2
        self.auto_coef = auto_coef
        self.max_iter = max_iter


        self.p_bests = self.particles
        self.p_bests_values = [self.fitness_function(i) for i in self.particles]
        self.g_best = self.p_bests[0]
        self.g_best_value = self.p_bests_values[0]
        self.update_bests()

        self.iter = 0
        self.is_running = True
        self.update_coef()

    def __str__(self):
        return f'[{self.iter}/{self.max_iter}] $w$:{self.w:.3f} - $c_1$:{self.c_1:.3f} - $c_2$:{self.c_2:.3f}'

    def next(self):
        if self.iter > 0:
            self.move_particles()
            self.update_bests()
            self.update_coef()

        self.iter += 1
        self.is_running = self.is_running and self.iter < self.max_iter
        return self.is_running

    def update_coef(self):
        if self.auto_coef:
            t = self.iter
            n = self.max_iter
            self.w = (0.4/n**2) * (t - n) ** 2 + 0.4
            self.c_1 = -3 * t / n + 3.5
            self.c_2 =  3 * t / n + 0.5

    def move_particles(self):

        # add inertia
        new_velocities = self.w * self.velocities
        # add cognitive component
        r_1 = np.random.random(self.N)
        r_1 = np.tile(r_1[:, None], (1, 20))
        new_velocities += self.c_1 * r_1 * (self.p_bests - self.particles)
        # add social component
        r_2 = np.random.random(self.N)
        r_2 = np.tile(r_2[:, None], (1, 20))
        g_best = np.tile(self.g_best[None], (self.N, 1))
        new_velocities += self.c_2 * r_2 * (g_best  - self.particles)

        self.is_running = np.sum(self.velocities - new_velocities) != 0

        # update positions and velocities
        self.velocities = new_velocities
        self.particles = self.particles + new_velocities


    def update_bests(self):
        fits = [self.fitness_function(i) for i in self.particles]

        for i in range(len(self.particles)):
            # update best personnal value (cognitive)
            if fits[i] < self.p_bests_values[i]:
                self.p_bests_values[i] = fits[i]
                self.p_bests[i] = self.particles[i]
                # update best global value (social)
                if fits[i] < self.g_best_value:
                    self.g_best_value = fits[i]
                    self.g_best = self.particles[i]


def griewank_func(x, n = 20):
    fr = 4000
    s = 0
    p = 1
    for j in range(n):
        s = s+x[j]**2
    for j in range(n):
        p = p*math.cos(x[j]/math.sqrt(j+1))
    return s/fr-p+1

num_particles = 30
num_features = 20

particles = np.random.uniform(-600,600, (num_particles, num_features))
velocities = (np.random.random((num_particles, num_features)) - 0.5) / 10

pso = PSO(particles, velocities, griewank_func)
i = 0
while pso.next():
    print("ITER: ", i, " pso: ", pso.p_bests_values[0])
    i += 1
