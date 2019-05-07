import argparse
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np   
import math
from scipy import stats
from scipy import spatial
from gensim.models import Word2Vec
'''
use existing matrix to run edge2vec
'''
def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run edge transition matrix.") 

    parser.add_argument('--input', nargs='?', default='weighted_graph.txt',
                        help='Input graph path')

    parser.add_argument('--matrix', nargs='?', default='matrix.txt',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='vector.txt',
                        help='Embeddings path') 
  
    parser.add_argument('--dimensions', type=int, default=128,
                        help='Number of dimensions. Default is 128.')

    parser.add_argument('--walk-length', type=int, default=3,
                        help='Length of walk per source. Default is 3.')

    parser.add_argument('--num-walks', type=int, default=2,
                        help='Number of walks per source. Default is 2.')

    parser.add_argument('--window-size', type=int, default=2,
                        help='Context size for optimization. Default is 2.')

    parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='weighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='directed', action='store_false')
    parser.set_defaults(directed=False)

    
    return parser.parse_args()
 
def read_graph(edgeList,weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(edgeList, nodetype=str, data=(('type',int),('weight',float),('id',int)), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(edgeList, nodetype=str,data=(('type',int),('id',int)), create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1.0

    if not directed:
        G = G.to_undirected()

    # print (G.edges(data = True))
    return G
 
def read_edge_type_matrix(file):
    '''
    load transition matrix
    '''
    matrix = np.loadtxt(file, delimiter=' ')
    return matrix


def simulate_walks(G, num_walks, walk_length,matrix,is_directed,p,q):
    '''
    generate random walk paths constrainted by transition matrix
    '''
    walks = []
    nodes = list(G.nodes())
    print 'Walk iteration:'
    for walk_iter in range(num_walks):
        print str(walk_iter+1), '/', str(num_walks)
        random.shuffle(nodes) 
        for node in nodes:
            # print "chosen node id: ",nodes
            walks.append(edge2vec_walk(G, walk_length, node,matrix,is_directed,p,q))  
    return walks

def edge2vec_walk(G, walk_length, start_node,matrix,is_directed,p,q): 
    # print "start node: ", type(start_node), start_node
    '''
    return a random walk path
    '''
    walk = [start_node]  
    while len(walk) < walk_length:# here we may need to consider some dead end issues
        cur = walk[-1]
        cur_nbrs =sorted(G.neighbors(cur)) #(G.neighbors(cur))
        random.shuffle(cur_nbrs)
        if len(cur_nbrs) > 0:
            if len(walk) == 1:
                rand = int(np.random.rand()*len(cur_nbrs))
                next =  cur_nbrs[rand]
                walk.append(next) 
            else:
                prev = walk[-2]
                pre_edge_type = G[prev][cur]['type']
                distance_sum = 0
                for neighbor in cur_nbrs:
                    neighbor_link = G[cur][neighbor] 
                    # print "neighbor_link: ",neighbor_link
                    neighbor_link_type = neighbor_link['type']
                    # print "neighbor_link_type: ",neighbor_link_type
                    neighbor_link_weight = neighbor_link['weight']
                    trans_weight = matrix[pre_edge_type-1][neighbor_link_type-1]
                    
                    if G.has_edge(neighbor,prev) or G.has_edge(prev,neighbor):#undirected graph
                        
                        distance_sum += trans_weight*neighbor_link_weight/p #+1 normalization
                    elif neighbor == prev: #decide whether it can random walk back
                        distance_sum += trans_weight*neighbor_link_weight
                    else:
                        distance_sum += trans_weight*neighbor_link_weight/q

                '''
                pick up the next step link
                ''' 

                rand = np.random.rand() * distance_sum
                threshold = 0 
                for neighbor in cur_nbrs:
                    neighbor_link = G[cur][neighbor] 
                    # print "neighbor_link: ",neighbor_link
                    neighbor_link_type = neighbor_link['type']
                    # print "neighbor_link_type: ",neighbor_link_type
                    neighbor_link_weight = neighbor_link['weight']
                    trans_weight = matrix[pre_edge_type-1][neighbor_link_type-1]
                    
                    if G.has_edge(neighbor,prev)or G.has_edge(prev,neighbor):#undirected graph
                        
                        threshold += trans_weight*neighbor_link_weight/p 
                        if threshold >= rand:
                            next = neighbor
                            break;
                    elif neighbor == prev:
                        threshold += trans_weight*neighbor_link_weight
                        if threshold >= rand:
                            next_link_end_node = neighbor
                            break;        
                    else:
                        threshold += trans_weight*neighbor_link_weight/q
                        if threshold >= rand:
                            next = neighbor
                            break;

                walk.append(next) 
        else:
            break #if only has 1 neighbour 
 
        # print "walk length: ",len(walk),walk
        # print "edge walk: ",len(edge_walk),edge_walk 
    return walk  
 

def main(args):  
    print "begin to read transition matrix"
    trans_matrix = read_edge_type_matrix(args.matrix)
    print trans_matrix

    print "------begin to read graph---------"
    G = read_graph(args.input,args.weighted,args.directed) 

    print "------begin to simulate walk---------" 
    walks = simulate_walks(G,args.num_walks, args.walk_length,trans_matrix,args.directed,args.p,args.q) 
    # print walks  
    model = Word2Vec(walks, size=args.dimensions, window=args.window_size, min_count=0, sg=1, workers=args.workers, iter=args.iter)
    model.wv.save_word2vec_format(args.output)
if __name__ == "__main__":
    args = parse_args()

    main(args)   
