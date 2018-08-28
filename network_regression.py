'''Code to perform the Network Regression.'''

import cvxpy
import json
import numpy as np
import networkx as nx
from kink_finder import get_kink_point

def gene_co_express(data,cutoff):
'''To construct the Gene Coexpression matrix, and remove values up to a certain percentage cutoff.'''
	W = np.corrcoef(data.T)
	W = W-np.eye(np.shape(W)[0])
	W = np.power(W,2)
	clip_val = np.percentile(W.flatten(),cutoff)
	W = np.clip(W-clip_val,0,np.inf)
	to_add = np.ceil(W)
	W += clip_val*to_add
	return W

def S_from_W(A,W):
'''To construct an S matrix, for the networked constraint, from the clipped coexpression matrix.'''
	G = nx.from_numpy_matrix(A)
	L = nx.normalized_laplacian_matrix(G).todense()
	D = np.sum(A,axis=1)
	D_clip = np.diag(np.clip(D, 0, 1))
	W = W+D_clip
	S = np.multiply(L,W)
	np.clip(S, -np.inf, 1)
	return S

def regress(genes,lambd,alpha,xs,ys,left,S):
'''To perform the regression using convex optimisation.'''
	cost = 0
	n_genes = np.shape(genes)[1]
	constr = []
	beta = cvxpy.Variable(n_genes)
	# to prevent beta becoming very large.
	constr.append(cvxpy.norm(beta)<=1)
	x0,y0,k1,k2 = get_kink_point(xs,ys)
	if left:
		filtered_genes = genes[ys>y0]
	else:
		filtered_genes = genes[ys<y0]
	for i,gene_set in enumerate(genes):
		cost += beta.T*gene_set
	#the log sum exp constraint
	cost -= np.shape(filtered_genes)[0]*cvxpy.log_sum_exp(filtered_genes*beta)
	# if a linear regression is being used, this allows S to be an empty matrix.
	if lambd>0.0:
		cost -= lambd*alpha*cvxpy.power(cvxpy.norm(beta),2)
		cost -= lambd*(1.0-alpha)*cvxpy.quad_form(beta,S)
	prob = cvxpy.Problem(cvxpy.Maximize(cost),constr)
	# a slightly increased tolerance (default is 1e-7) to reduce run times
	a = prob.solve(solver=cvxpy.SCS,eps=1e-5)
	return beta.value

def add_network_regression(filename, lambd, alpha, cutoff):
'''To add the gene coexpression matrix, regression parameters and network constraint matrix to the existing data file.
For a given file, value of lambda, alpha and cutoff percentage.'''
	data = json.load(open(filename))
	pareto = data['pareto']
	ys = np.array(list(map(lambda i : i['obj1'],pareto)))
	xs = np.array(list(map(lambda i : i['obj2'],pareto)))
	genes = np.array(list(map(lambda i : i['gene_set'],pareto)))
	W = gene_co_express(genes,cutoff)
	A = np.ceil(W)
	S = S_from_W(A,W)
	beta1 = regress(genes,lambd,alpha,xs,ys,True,S).flatten()
	beta2 = regress(genes,lambd,alpha,xs,ys,False,S).flatten()
	to_save = {'beta1':beta1.tolist(),'beta2':beta2.tolist(),'A':A.tolist(),'S':S.tolist()}
	data['network'] = to_save
	with open(filename, 'w') as outfile:
	    json.dump(data, outfile)
	return beta1,beta2,W,A,S
