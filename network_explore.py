'''To create a reduced network from a network.'''

def gen_sorted_node_list(G):
	node_dict = dict(G.degree)
	to_return = sorted(node_dict,key=node_dict.get,reverse=True)
	return to_return

def reduce_graph(graphs):
	key_nodes = []
	for graph in graphs:
		sorted_nodes = gen_sorted_node_list(graph)
		accounted_for = set()
		nodes = set(graph.nodes)
		while accounted_for!=nodes:
			node = sorted_nodes.pop(0)
			key_nodes.append(node)
			neighbours = set(graph[node].keys())
			accounted_for.update(neighbours)
			accounted_for.update([node])
			graph.remove_node(node)
			sorted_nodes = gen_sorted_node_list(graph)
	return key_nodes
