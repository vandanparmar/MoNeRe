import numpy as np
import json
import pareto
from tqdm import tqdm
from itertools import repeat
import networkx as nx
import kink_finder
import network_explore
import multiprocessing

def iqr(x):
'''For calculating the interquartile range.'''
	q75, q25 = np.percentile(x, [75 ,25])
	iqri = q75 - q25
	return iqri

def prep_fba_set(points, descriptor):
'''To add string descriptors to a set of Pareto points, used when constructing the overall Pareto front from a variety of reconstructions.'''
	to_return = list(zip(points[:,0].tolist(),points[:,1].tolist(),repeat(descriptor)))
	return to_return

def find_optimal(fba_points):
'''To construct the Pareto front from the set of reconstructed points.'''
	fba_points.sort(key = lambda x : x[0],reverse = True)
	to_return = []
	current_max = -np.inf
	for point in fba_points:
		if point[1] > current_max:
			current_max = point[1]
			to_return.append(point)
	return to_return

def eval_pareto(model ,obj1 ,obj2 ,gene_set,cores):
'''To evaluate the fluxes for reconstructed points.'''
	bounds = np.array(list(map(lambda reaction : reaction.bounds, model.reactions)))
	lb = bounds[:,0]
	lb = np.clip(lb, a_min=-np.inf,a_max = np.inf)
	ub = bounds[:,1]
	ub = np.clip(ub,a_min= -np.inf,  a_max=np.inf)
	gene_dict = {x.id:i for i, x in enumerate(model.genes)}
	condts = pareto.gene_condts(model, gene_dict)
	if cores==0:
		evaluate = pareto.evaluate
		gene_set = tqdm(gene_set)
		to_return = list(map(lambda i : evaluate(i, obj1, obj2, model, condts, lb, ub), gene_set))
# For parallelisation
	else:
		evaluate = pareto.eval_wrapper
		pool = multiprocessing.Pool(cores)	
		args = zip(gene_set, repeat(obj1),repeat(obj2),repeat(model),repeat(condts),repeat(lb),repeat(ub))
		to_return = list(tqdm(pool.imap(evaluate, args,chunksize=50),total = len(gene_set)))
	return np.array(to_return)

def create_network_vals(key_nodes,G,maxes,recon_points,low,high,pareto_genes):
'''To create the significant node values given the reduced network.'''
	node_dict = {v:i for i,v in enumerate(maxes)}
	key_node_dict = {v:i for i,v in enumerate(key_nodes)}	
	small_indices = np.random.random_integers(low, high, size=(recon_points,len(key_nodes)))
	indices = np.zeros((recon_points,len(maxes)))	
	for i,maxi in enumerate(maxes):
		if node_dict[maxi] in key_nodes:
			indices[:,i] = small_indices[:,key_node_dict[node_dict[maxi]]]
		else:
			neighbours = list(G[node_dict[maxi]].keys())
			neighbours = [x for x in neighbours if x in key_nodes]
			neighbours = list(map(lambda i : key_node_dict[i],neighbours))
			choices = np.random.choice(neighbours, recon_points)
			indices[:,i] = small_indices[np.arange(recon_points),choices]
	indices = np.array(indices, dtype=np.int)
	to_set = pareto_genes[indices,maxes]
	return to_set

def reconstruct(filename,recon_points,h_p,obj1,obj2,cores=0):
'''To reconstruct a Pareto front given a file with an existing network regression.'''
	data = json.load(open(filename))
	network = data['network']
	pareto_data = data['pareto']
	y_plot = np.array(list(map(lambda i : i['obj1'], pareto_data)))
	x_plot = np.array(list(map(lambda i : i['obj2'], pareto_data)))
	pareto_genes = np.array(list(map(lambda i : i['gene_set'], pareto_data)))

	bounds = np.array(list(map(lambda reaction : reaction.bounds, h_p.reactions)))

	x0,y0,k1,k2 = kink_finder.get_kink_point(x_plot, y_plot)
	phase_trans = np.abs(x_plot-x0).argmin()

	beta1 = np.array(network['beta1']).flatten()
	beta2 = np.array(network['beta2']).flatten()

	iqr1 = iqr(beta1)
	iqr2 = iqr(beta2)

	mean1 = np.mean(beta1)
	mean2 = np.mean(beta2)

	nodes1 = np.shape(beta1[beta1>mean1+1.0*iqr1])[0]
	nodes2 = np.shape(beta2[beta2>mean2+1.0*iqr2])[0]

	maxes1 = np.argpartition(beta1, -nodes1)[-nodes1:]
	maxes2 = np.argpartition(beta2, -nodes2)[-nodes2:]

	n_genes = len(h_p.genes)
	to_pareto = np.random.uniform(low=0.0, high=2.0, size=(recon_points*2, n_genes))
	to_pareto_noise = np.random.uniform(low=0.0, high=2.0, size=(recon_points,n_genes))

	A = np.array(network['A'])

	small1 = A[maxes1][:,maxes1]
	small2 = A[maxes2][:,maxes2]

	G1 = nx.from_numpy_matrix(small1)
	G2 = nx.from_numpy_matrix(small2)

	graphs1 = list(nx.connected_component_subgraphs(G1))
	graphs2 = list(nx.connected_component_subgraphs(G2))

	key_nodes1 = network_explore.reduce_graph(graphs1)
	key_nodes2 = network_explore.reduce_graph(graphs2)

	G1 = nx.from_numpy_matrix(small1)
	G2 = nx.from_numpy_matrix(small2)

	to_set_1 = create_network_vals(key_nodes1, G1, maxes1, recon_points, low=0, high=phase_trans, pareto_genes=pareto_genes)
	to_set_2 = create_network_vals(key_nodes2, G2, maxes2, recon_points, low=phase_trans+1,high=len(x_plot)-1, pareto_genes=pareto_genes)
	to_pareto[::2, maxes1] = to_set_1
	to_pareto[1::2, maxes2] = to_set_2

	pareto_new = eval_pareto(h_p, obj1, obj2, to_pareto,cores)

	for i,x in enumerate(bounds):
		h_p.reactions[i].bounds = x

	pareto_noise = eval_pareto(h_p,obj1,obj2,to_pareto_noise,cores)

	for i,x in enumerate(bounds):
		h_p.reactions[i].bounds = x

	pareto_left = pareto_new[::2]
	pareto_right = pareto_new[1::2]

	paretos = prep_fba_set(pareto_left, 'left')
	paretos.extend(prep_fba_set(pareto_right,'right'))
	paretos.extend(prep_fba_set(pareto_noise,'noise'))

	pareto_collection = find_optimal(paretos)
	pareto_y = list(map(lambda x : x[0],pareto_collection))
	pareto_x = list(map(lambda x : x[1],pareto_collection))

	to_add = {'pareto_left':pareto_left.tolist(),
		'pareto_right':pareto_right.tolist(),
		'pareto_noise':pareto_noise.tolist(),
		'pareto_y':pareto_y,
		'pareto_x':pareto_x }
	data['recon'] = to_add
	with open(filename, 'w') as outfile:
	    json.dump(data, outfile)

	return pareto_left,pareto_right,pareto_noise,pareto_y,pareto_x

