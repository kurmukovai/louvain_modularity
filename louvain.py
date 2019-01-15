import numpy as np

def modularity(A, partition, silent=True):
    n = A.shape[0]
    k = A.sum(axis=1)
    Q = 0
    m = A.sum() / 2

    for i in range(n):
        for j in range(n):
            if partition[i] == partition[j]:
                dQ = A[i,j] - k[i]*k[j] / 2 / m
                if not silent:
                    print('[{}, {}] = {} - {}*{}/{} dQ = {}'.format(i,j, A[i,j], k[i], k[j], 2*m, dQ))
                Q += dQ
    return Q / 2 / m


def Q_x(A, partition, node, cluster, m, degrees):
    # Modularity gain = remove_x() + add_x()
#     m = A.sum() / 2
#     partition = labels.copy()
    stored_value = partition[node]
    partition[node] = cluster
    cluster_nodes = np.where(partition==cluster)[0]

    Q_x_cluster = (1 / 2 / m) * (A[node, cluster_nodes].sum() - (degrees[node] / 2 / m) * A[cluster_nodes, :].sum())
    Q_xx = (degrees[node] / 2 / m)**2
    
    partition[node] = stored_value
    
    return 2 * (Q_x_cluster + Q_xx)

def add_x(A, labels, node, cluster, m, degrees):
    #print('add_x')
    return Q_x(A, labels, node, cluster, m, degrees)

def remove_x(A, labels, node, cluster, m, degrees):
    #print('remove_x')
    return - Q_x(A, labels, node, cluster, m, degrees)

def louvain_assign(A, init_partition=None, random_state=803):
    '''
    Louvain algorithm for a single graph assign step
    '''
    
    random = np.random.RandomState(random_state)
    n_nodes = A.shape[0]
    if init_partition is None:
        init_partition = np.arange(n_nodes)

    degrees = A.sum(axis=1)
    m = degrees.sum() / 2
    partition = init_partition.copy()
    order = random.permutation(np.arange(n_nodes)) # traversal order
    best_gain = 1e-9 # <---- interesting idea to threshold modularity gain
    cluster_weights = degrees.copy()
    
    while best_gain > 1e-12:
        all_gain = []
        for node in order:

            best_community = partition[node]
            Q_loss = remove_x(A, partition, node, best_community, m, degrees)
            neighbours = np.where(A[node, :]>0)[0]
            neigh_communities = np.unique(partition[neighbours])
            Q_total = []
            
            for community in neigh_communities: 
                Q_gain = add_x(A, partition, node, community, m, degrees)
                Q_total.append(Q_gain)

            Q_total = np.array(Q_total) + Q_loss
            best_Q_idx = np.argmax(Q_total)
            best_community = neigh_communities[best_Q_idx]

            current_gain = Q_total[best_Q_idx]
            all_gain.append(current_gain)
            partition[node] = best_community
           
        best_gain = np.max(all_gain)
    return partition

def gen_mapping(A, partition):
    
    n_clusters = np.unique(partition).shape[0]
    mapping = dict(zip(np.unique(partition), np.arange(n_clusters)))
    labels = [mapping[key] for key in partition]
    n_nodes = A.shape[0]
    
    assignment = np.zeros((n_nodes, n_clusters))
    for i, label in enumerate(labels):
        assignment[i, label] = 1
        
    return assignment, labels

def louvain_combine(A, partition):
    C, old_labels = gen_mapping(A, partition)
    new_A = C.T.dot(A).dot(C)
    np.fill_diagonal(new_A, 0)
    return new_A, np.array(old_labels)

def relabel(levels):

    full_labels = []
    full_labels.append(levels[0])
    
    for i,labels in enumerate(levels[1:]): 
        temp = []
        for j,elem in enumerate(full_labels[i]):
            temp.append(labels[elem])
        full_labels.append(np.array(temp))
    return full_labels

def louvain(A, init_partition=None, random_state=803):
    
    _A = A.copy()
    all_levels = []
    current_partition = np.arange(A.shape[0])
    new_partition = louvain_assign(_A, init_partition=init_partition, random_state=random_state)
    
    while True:
        
        _A, new_partition = louvain_combine(_A, new_partition)
        all_levels.append(new_partition)
        if _A.shape[0] == 1:
            break
        new_partition = louvain_assign(_A, random_state=random_state)

    
    levels = relabel(all_levels)
    
    return levels
