'''Based off code by Marco Barsacchi, Claudio Angione, but edited and added to. 
For dealing with all things Pareto related, plotting, creating and saving.
'''

import multiprocessing
import time
from itertools import repeat
from collections import Sequence
import kink_finder
from matplotlib import pyplot as plt
import numpy as np
from deap import tools, base, creator

def h(y):
'''The lazy step function used to relate gene expression value to the scaling factor of flux bounds.'''
	return np.power((1+ np.abs(np.log(y))),np.sign(y-1))


def gene_condts(model,gene_dict):
'''To convert the gene names to more easily readable format. Lists of lists, inner lists are minimised and outer lists are maximised.'''
	condts = []
	for reaction in model.reactions:
		condt = reaction.gene_reaction_rule.replace('(','').replace(')','')
		if condt != '':
			condt = condt.split(" or ")
			condt = list(map(lambda i : i.split(" and "),condt))
			condt = list(map( lambda genes : list(map(lambda gene: gene_dict[gene],genes)), condt ))
			condts.append(condt)
		else:
			condt = [[np.inf]]
			condts.append(condt)
	return condts

def set_weights(individual,model,condts,lb,ub):
'''To apply the gene expression data to change the flux bounds of reactions in the model.'''
	def set_bounds(arg):
		bound, reaction = arg
		reaction.bounds = (bound[0],bound[1])
		return reaction
	# individual values replaced by values from genetic alg, inner lists are minimised, outer lists are maximised, giving individual expression values
	condts = list(map(lambda condt: np.max(list(map(lambda ors: np.min( list(map(lambda i: 1.0 if (i==np.inf) else individual[i],ors))),condt))),condts))
	condts = np.array(condts)
	hs = h(condts)
	lower = np.multiply(hs,lb)
	upper = np.multiply(hs,ub)
	bounds = zip(lower,upper)
	model.reactions = list(map(set_bounds ,zip(bounds,model.reactions)))
	return model

def mutPolynomialGenes(individual,eta,low,high,indpb):
	"""Modified polynomial mutation from the original NSGA-II"""	
	size = len(individual)
	if not isinstance(low,Sequence):
		low = repeat(low)
	elif len(low)< size:
		raise IndexError("Low is too short")
	if not isinstance(high,Sequence):
		low = repeat(high)
	elif len(low)< size:
		raise IndexError("High is too short")

	individual = np.array(individual)
	rands = np.random.rand(size)
	pm_rand = 2.0*np.random.randint(2,size=size)-1.0
	mut_power = np.power(eta+1.0,-1)
	delta_q = np.power(rands,mut_power)
	delta_q = np.multiply(delta_q,pm_rand)
	individual += delta_q
	individual = list(np.clip(individual,a_min=low,a_max=high))
	return individual

	


def long_evaluate(individual,obj1,obj2,model,condts,lb,ub):
'''To evaluate but return fluxes of all reactions, not recommended, very slow.'''
	this_model = model
	this_model = set_weights(individual,this_model,condts,lb,ub)

	this_model.objective = this_model.problem.Objective(obj1,direction='max')

	sol1 = this_model.optimize()
	flux1 = sol1.objective_value

	flux1_constr = this_model.problem.Constraint(obj1,lb=flux1,ub=flux1)

	this_model.add_cons_vars(flux1_constr)

	this_model.objective = this_model.problem.Objective(obj2,direction='max')

	sol2 = this_model.optimize()
	fluxes2 = sol2.fluxes

	this_model.remove_cons_vars(flux1_constr)

	return fluxes2

def eval_wrapper(args):
'''Used for Pareto reconstruction.'''
	individual,obj1,obj2,model,condts,lb,ub = args
	f1,f2 = evaluate(individual,obj1,obj2,model,condts,lb,ub)
	return f1,f2


def evaluate(individual,obj1,obj2,model,condts,lb,ub):
'''To evaluate the fluxes through reactions. Used by the main Pareto loop.'''
	this_model = model
	this_model.solver.configuration.tolerances.optimality = 1e-4
	this_model.solver.configuration.tolerances.feasibility = 1e-4
	this_model.solver.configuration.timeout = 10
	
	this_model = set_weights(individual,this_model,condts,lb,ub)

	this_model.objective = this_model.problem.Objective(obj1,direction='max')

	flux1 = this_model.slim_optimize()

	if (np.isnan(flux1)):
		return -np.inf,-np.inf
	else:
		flux1_constr = this_model.problem.Constraint(obj1,lb=flux1,ub=flux1)

		this_model.add_cons_vars(flux1_constr)

		this_model.objective = this_model.problem.Objective(obj2,direction='max')

		flux2 = this_model.slim_optimize()

		this_model.remove_cons_vars(flux1_constr)

		return flux1,flux2

def create_algo_holder(toolbox,model,obj1,obj2,cores):
'''To create the DEAP toolbox with all the necessary parameters and functions.'''
	n_genes = len(model.genes)
	def init_attribute():
		return 1. + 2.*(np.random.uniform()-0.5) 

	bounds = np.array(list(map(lambda reaction : reaction.bounds, model.reactions)))
	lb = bounds[:,0]
	lb = np.clip(lb, a_min=-100.0,a_max = 100.0)
	ub = bounds[:,1]
	ub = np.clip(ub,a_min= -100.0,  a_max=100.0)
	gene_dict = {x.id:i for i,x in enumerate(model.genes)}
	condts = gene_condts(model,gene_dict)
	#initial max and min gene expressions
	min_v = list(np.zeros((n_genes,)))
	max_v = list(np.zeros((n_genes,))+100.0)
	#for parallelisation
	if (cores!=0):
		pool = multiprocessing.Pool(cores)
		toolbox.register("map",pool.map)
	toolbox.register("init_val", init_attribute)
	toolbox.register("individual", tools.initRepeat, creator.Individual,
					 toolbox.init_val, n=n_genes)
	toolbox.register("population", tools.initRepeat, list, toolbox.individual)
	toolbox.register("mate", tools.cxSimulatedBinaryBounded, eta=20, low=min_v, up=max_v)
	toolbox.register("mutate", mutPolynomialGenes, eta= 20, indpb=0.5, low=min_v, high=max_v)
	toolbox.register("select", tools.selRandom)
	toolbox.register("evaluate", evaluate,obj1 = obj1, obj2=obj2,model = model,condts = condts,lb=lb,ub=ub)
	return toolbox


def plot_pareto(pareto,filename,xlabel,ylabel,figure_str):
'''For plotting Pareto fronts. Mostly used in batch code.'''
	y_plot = np.array(list(map(lambda i : i['obj1'],pareto)))
	x_plot = np.array(list(map(lambda i : i['obj2'],pareto)))
	plt.clf()
	plt.plot(x_plot,y_plot,c='b', label = 'Pareto Points',marker='.',linestyle='None')
	try:
		x0,y0,k1,k2 = kink_finder.get_kink_point(x_plot,y_plot)
		x_plot = np.sort(np.append(x_plot,x0))
		plt.plot(x_plot,kink_finder.piecewise_linear(x_plot,x0,y0,k1,k2),c='lawngreen', label = 'Fitted Line')
		plt.plot(x0,y0,'*',c='r', label = 'Phase Transition Point')
	except:
		print('No Phase Transition')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.legend()
	plt.savefig(figure_str+filename+'.png')
	plt.savefig(figure_str+filename+'.eps')
	plt.clf()


def pareto(generations,pop_size,model, obj1, obj2, batch = False,cores=0):
'''The main Pareto front creation function.'''
	creator.create("FitnessMax",base.Fitness,weights=(1.0,1.0))
	creator.create("Individual",list, fitness=creator.FitnessMax)

	pops = []
	vals = []

	#setting up toolbox parameters
	nsga2_holder = base.Toolbox()
	nsga2_holder = create_algo_holder(nsga2_holder,model,obj1,obj2,cores)

	stats_fit = tools.Statistics(key=lambda ind: ind.fitness.values)
	stats_size = tools.Statistics(key=len)

	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
	mstats.register("avg", np.mean, axis=0)
	mstats.register("std", np.std, axis=0)
	mstats.register("min", np.min, axis=0)
	mstats.register("max", np.max, axis=0)
	logbook = tools.Logbook()
	logbook.header = "gen", "fitness", "hypervolume", "time"
	logbook.chapters["fitness"].header = "min", "max"

	pop = nsga2_holder.population(n=pop_size)
	CXPB, MUTPB = 0.9,0.1
	pareto = tools.ParetoFront()

	# initialise population with fitnesses
	fitnesses = nsga2_holder.map(nsga2_holder.evaluate, pop)
	for ind, fit in zip(pop, fitnesses):
		ind.fitness.values = fit

	pop = tools.selNSGA2(pop,pop_size)
	
	#the main Pareto loop
	for gen in range(generations):
		t_start = time.time()
		offspring = nsga2_holder.select(pop,pop_size)
		offspring = list(map(nsga2_holder.clone,offspring))
		for c1,c2 in zip(offspring[::2],offspring[1::2]):
			val = np.random.rand()
			if val<CXPB:
				nsga2_holder.mate(c1,c2)
				del c1.fitness.values
				del c2.fitness.values
			elif val<CXPB+MUTPB:
				nsga2_holder.mutate(c1)
				nsga2_holder.mutate(c2)
				del c1.fitness.values
				del c2.fitness.values

		#create a list of items without fitnesses
		invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
		fitnesses = nsga2_holder.map(nsga2_holder.evaluate, invalid_ind)
		for i,(ind,fit) in enumerate(zip(invalid_ind, fitnesses)):
			ind.fitness.values = fit
		pop = tools.selNSGA2(pop+offspring,pop_size)

		record = mstats.compile(pop)
		eval_time = time.time()-t_start
		logbook.record(gen=gen, time=eval_time, **record)
		#for use with TQDM
		if batch is False:
			print(logbook.stream)
		else:
			batch(logbook.stream)

		pareto.update(pop)
		val = list(zip(*[p.fitness.values for p in pareto]))
		vals.append(val)
		pops.append(pop)

	return pops,vals,pareto