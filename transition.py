import argparse
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np   
import math
from scipy import stats

'''
first version: unweighted, undirected network
use edge random walk to generate edge transction matrix based on EM algorithm
'''
def parse_args():
    '''
    Parses the node2vec arguments.
    '''
    parser = argparse.ArgumentParser(description="Run edge transition matrix.") 

    parser.add_argument('--input', nargs='?', default='data.csv',
                        help='Input graph path')

    parser.add_argument('--output', nargs='?', default='matrix.txt',
                        help='store transition matrix')
    
    parser.add_argument('--type_size', type=int, default=3,
                        help='Number of edge types. Default is 3.')

    parser.add_argument('--em_iteration', default=5, type=int,
                      help='EM iterations for transition matrix')

    parser.add_argument('--e_step', default=3, type=int,
                      help='E step in the EM algorithm: there are four expectation metrics')
    
    parser.add_argument('--dimensions', type=int, default=10,
                        help='Number of dimensions. Default is 10.')

    parser.add_argument('--walk-length', type=int, default=3,
                        help='Length of walk per source. Default is 3.')

    parser.add_argument('--num-walks', type=int, default=2,
                        help='Number of walks per source. Default is 2.')

    parser.add_argument('--window-size', type=int, default=5,
                        help='Context size for optimization. Default is 5.')

    parser.add_argument('--iter', default=10, type=int,
                      help='Number of epochs in SGD')

    parser.add_argument('--workers', type=int, default=8,
                        help='Number of parallel workers. Default is 8.')

    parser.add_argument('--p', type=float, default=1,
                        help='Return hyperparameter. Default is 1.')

    parser.add_argument('--q', type=float, default=1,
                        help='Inout hyperparameter. Default is 1.')

    parser.add_argument('--weighted', dest='weighted', action='store_true',
                        help='Boolean specifying (un)weighted. Default is unweighted.')
    parser.add_argument('--unweighted', dest='unweighted', action='store_false')
    parser.set_defaults(weighted=False)

    parser.add_argument('--directed', dest='directed', action='store_true',
                        help='Graph is (un)directed. Default is undirected.')
    parser.add_argument('--undirected', dest='undirected', action='store_false')
    parser.set_defaults(directed=False)

    return parser.parse_args()
 
def read_graph(edgeList,weighted=False, directed=False):
    '''
    Reads the input network in networkx.
    '''
    if weighted:
        G = nx.read_edgelist(edgeList, nodetype=str, data=(('type',int),('id',int)), create_using=nx.DiGraph())
    else:
        G = nx.read_edgelist(edgeList, nodetype=str,data=(('type',int),('id',int)), create_using=nx.DiGraph())
        for edge in G.edges():
            G[edge[0]][edge[1]]['weight'] = 1

    if not directed:
        G = G.to_undirected()

    # print (G.edges(data = True))
    return G
 
def initialize_edge_type_matrix(type_num):
    initialized_val = 1.0/(type_num*type_num)
    matrix = [ [ initialized_val for i in range(type_num) ] for j in range(type_num) ]
    return matrix


def simulate_walks(G, num_walks, walk_length,matrix):
    walks = []
    links = list(G.edges(data = True))
    print 'Walk iteration:'
    for walk_iter in range(num_walks):
        print str(walk_iter+1), '/', str(num_walks)
        random.shuffle(links)
        count = 1000
        for link in links:
            # print "chosen link id: ",link[2]['id']
            walks.append(edge2vec_walk(G, walk_length, link,matrix)) 
            count = count - 1
            if count == 0 and len(links)>1000:#control the pairwise list length
                break
    return walks

def edge2vec_walk(G, walk_length, start_link,matrix): 
    # print "start link: ", type(start_link), start_link
    walk = [start_link] 
    result = [str(start_link[2]['type'])]
    # print "result ",result
    while len(walk) < walk_length:# here we may need to consider some dead end issues
        cur = walk[-1]
        start_node = cur[0]
        end_node = cur[1]
        cur_edge_type = cur[2]['type']

        '''
        find the direction of link to go. If a node degree is 1, it means if go that direction, there is no other links to go further
        if the link are the only link for both nodes, the link will have no neighbours (need to have teleportation later)
        '''
        '''
        consider the hub nodes and reduce the hub influence
        '''
        start_direction = 0
        end_direction = 0
        if G.degree(start_node) > 1:
            start_direction = 1.0/G.degree(start_node)
        if G.degree(end_node) > 1: 
            end_direction = 1.0/G.degree(end_node)
        prob = start_direction/(start_direction+end_direction)
        # print "start node: ", start_node, " degree: ", G.degree(start_node)
        # print "end node: ", end_node, " degree: ", G.degree(end_node)

        # print cur[0], cur[1]
        rand = np.random.rand() 
        # print "random number ",rand
        # print "probability for start node: ",prob

        if prob >= rand:
            # print "yes"
            direction_node = start_node
            left_node = end_node
        else:
            direction_node = end_node
            left_node = start_node
        # print "directed node: ",direction_node
        # print "left_node node: ",left_node
        '''
        here to choose which link goes to. There are three conditions for the link based on node distance. 0,1,2
        '''
        neighbors = G.neighbors(direction_node) 
        # print G.has_edge(1,3)
        # print G.has_edge(3,1)
        '''
        calculate sum of distance, with +1 normalization
        '''
        distance_sum = 0
        for neighbor in neighbors:
            # print "neighbors:", neighbor
            neighbor_link = G[neighbor][direction_node]#get candidate link's type
            # print "neighbor_link: ",neighbor_link
            neighbor_link_type = neighbor_link['type']
            # print "neighbor_link_type: ",neighbor_link_type
            trans_weight = matrix[cur_edge_type-1][neighbor_link_type-1]
            if G.has_edge(neighbor,left_node):#undirected graph
                
                distance_sum += trans_weight*1.0/(1+1) #+1 normalization
            # elif neighbor == left_node: #decide whether it can random walk back
            #     distance_sum += 1.0/(0+1)
            else:
                distance_sum += trans_weight*1.0/(1+2)

        '''
        pick up the next step link
        '''
        # random.shuffle(neighbors)
        rand = np.random.rand() * distance_sum
        threshold = 0
        # next_link_end_node = 0 
        neighbors2 = G.neighbors(direction_node) 
        for neighbor in neighbors2:
            # print "current threshold: ", threshold
            neighbor_link = G[neighbor][direction_node]#get candidate link's type
            neighbor_link_type = neighbor_link['type']
            trans_weight = matrix[cur_edge_type-1][neighbor_link_type-1]
            if G.has_edge(neighbor,left_node):#undirected graph
                threshold += trans_weight*1.0/(1+1) #+1 normalization
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break;
            # elif neighbor == left_node:
            #     distance_sum += 1.0/(0+1)
            else:
                threshold += trans_weight*1.0/(1+2)
                if threshold >= rand:
                    next_link_end_node = neighbor
                    break;

        # print "distance_sum: ",distance_sum
        # print "rand: ", rand, " threshold: ", threshold
        # print "next_link_end_node: ",next_link_end_node
        next_link = G[direction_node][next_link_end_node]
        # next_link = G.get_edge_data(direction_node,next_link_end_node)
        
        next_link_tuple = tuple()
        next_link_tuple += (direction_node,)
        next_link_tuple += (next_link_end_node,)
        next_link_tuple += (next_link,)
        # print type(next_link_tuple)
        # print next_link_tuple
        walk.append(next_link_tuple)
        result.append(str(next_link_tuple[2]['type']))
        # print "walk length: ",len(walk),walk
    # print "path: ",result
    return result  

'''
E step, update transition matrix
'''
def update_trans_matrix(walks,type_size,evaluation_metric):
    #here need to use list of list to store all edge type numbers and use KL divergence to update
    matrix = [ [ 0 for i in range(type_size) ] for j in range(type_size) ]
    repo = dict()
    for i in range(type_size):#initialize empty list to hold edge type vectors
        repo[i] = []

    for walk in walks:
        curr_repo = dict()#store each type number in current walk
        for edge in walk:
            edge_id = int(edge) - 1 
            if edge_id in curr_repo:
                curr_repo[edge_id] = curr_repo[edge_id]+1
            else:
                curr_repo[edge_id] = 1

        for i in range(type_size):
            
            # print "curr_repo[i]: ",curr_repo[i],type(curr_repo[i])
            if i in curr_repo:
                repo[i].append(curr_repo[i]) 
            else:
                repo[i].append(0) 
    
    for i in range(type_size):
        # print "repo ",i, ": ",repo[i],type(repo[i])
        for j in range(type_size):  
            if evaluation_metric == 1:
                sim_score = wilcoxon_test(repo[i],repo[j])  
                matrix[i][j] = sim_score
                # print "each pair of edge type sim_score: ", sim_score
            elif evaluation_metric == 2:
                sim_score = entroy_test(repo[i],repo[j])  
                matrix[i][j] = sim_score
            elif evaluation_metric == 3:
                sim_score = spearmanr_test(repo[i],repo[j])  
                matrix[i][j] = sim_score
            elif evaluation_metric == 4:
                sim_score = pearsonr_test(repo[i],repo[j])  
                matrix[i][j] = sim_score 
            else:
                raise ValueError('not correct evaluation metric! You need to choose from 1-4')  

    return matrix

#pairwised judgement
def wilcoxon_test(v1,v2):# original metric: the smaller the more similar 
    result = stats.wilcoxon(v1, v2).statistic
    return 1/(math.sqrt(result)+1)

def entroy_test(v1,v2):#original metric: the smaller the more similar
    result = stats.entropy(v1,v2)
    return result

def spearmanr_test(v1,v2):#original metric: the larger the more similar 
    result = stats.mstats.spearmanr(v1,v2).correlation
    return sigmoid(result)

def pearsonr_test(v1,v2):#original metric: the larger the more similar
    result = stats.mstats.pearsonr(v1,v2)[0]
    return sigmoid(result)

def cos_test(v1,v2):
    return 1 - spatial.distance.cosine(v1, v2)

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def standardization(x):
    return (x+1)/2

def relu(x):
    return (abs(x) + x) / 2
    
def main(args): 
    # print "------begin to write graph---------"
    # generate_graph_write_edgelist(args.m1,args.m2,args.input)
    print "begin to initialize transition matrix"
    trans_matrix = initialize_edge_type_matrix(args.type_size)
    print trans_matrix
    print "------begin to read graph---------"
    G = read_graph(args.input,True,False)
    # print G.edges(data=True)
    # G=nx.barbell_graph(17,1)
    # draw_graph(G) 
    print "------begin to simulate walk---------"
    for i in range(args.em_iteration):
        walks = simulate_walks(G,args.num_walks, args.walk_length,trans_matrix)#M step
        print str(i), "th iteration for Upating transition matrix!"
        trans_matrix = update_trans_matrix(walks,args.type_size,args.e_step)#E step
        print "trans_matrix: ",trans_matrix
    # print walks 
    print "------finish!---------"
    np.savetxt(args.output, trans_matrix)
if __name__ == "__main__":
    args = parse_args()

    main(args)   
