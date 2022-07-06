import numpy as np
import networkx as nx
import pickle
import scipy.sparse as sp
from load_data.data_params import get_dataInfo
import time

dataname, file_path, unit_time, _, _, m = get_dataInfo()
file_path = '.' + file_path + dataname + '_'
print(file_path)
def u2idx(nodes):
    return {node:i+1 for i,node in enumerate(nodes)}

def obtain_user_dict(file_path):
    """ create a map: user->id, i.e., reindex userid. """
    try:
        with open(file_path+"/u2idx.pkl", 'rb') as f:
            u2idx_dict = pickle.load(f)
    except:
        train_file = file_path + "/cascade_train.txt"
        val_file = file_path + "/cascade_validation.txt"
        test_file = file_path + "/cascade_test.txt"
        nodes = set()
        with open(train_file, 'r') as casf:
            for line in casf:
                parts = line.strip().split('\t')
                edges = parts[-2].split(" ")
                for edge in edges:
                    edge = edge.split(":")
                    nodes.add(edge[0])
                    nodes.add(edge[1])
        with open(val_file, 'r') as casf:
            for line in casf:
                parts = line.strip().split('\t')
                edges = parts[-2].split(" ")
                for edge in edges:
                    edge = edge.split(":")
                    nodes.add(edge[0])
                    nodes.add(edge[1])
        with open(test_file, 'r') as casf:
            for line in casf:
                parts = line.strip().split('\t')
                edges = parts[-2].split(" ")
                for edge in edges:
                    edge = edge.split(":")
                    nodes.add(edge[0])
                    nodes.add(edge[1])
        nodes = list(nodes)
        u2idx_dict = u2idx(nodes)
        with open(file_path+"/u2idx.pkl", 'wb') as f:
            pickle.dump(u2idx_dict, f)

    return u2idx_dict

def obtain_N():
    u2idx_dict = obtain_user_dict(file_path)
    return len(u2idx_dict)

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return sp.eye(len(rowsum)) - mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def load_cascade(file_path, file_type=0):
    # 0 train, 1 val, 2 test
    if file_type == 0:
        file_name = file_path + "/cascade_shortestpath_train.txt"
    elif file_type == 1:
        file_name = file_path + "/cascade_shortestpath_validation.txt"
    else:
        file_name = file_path + "/cascade_shortestpath_test.txt"
    user_dict = obtain_user_dict(file_path)

    # labels = []
    cascades_info = {}

    with open(file_name, 'r') as casf:
        for line in casf:
            cascade_lst = []
            timespan_lst = []
            # node_time = {}
            g = nx.DiGraph()
            # cascade id, origin nodes, edges, labels
            parts = line.strip().split('\t')
            # add label
            label = np.log(int(parts[-1]) + 1) / np.log(2.0)
            cascade_id = parts[0]

            edges = parts[1:-1]
            for i, edge in enumerate(edges):
                edge = edge.split(":")
                if i == 0:
                    cascade_lst.append(user_dict[edge[0]])
                    # convert to unit time
                    timespan_lst.append(float(edge[-1]) / unit_time + 1)
                    continue
                time = float(edge[-1]) / unit_time + 1
                edge = edge[0].split(',')

                cascade_lst.append(user_dict[edge[-1]])
                timespan_lst.append(time)
                for i in range(len(edge) - 1):
                    g.add_edge(user_dict[edge[i]], user_dict[edge[i+1]])

            adj = nx.adj_matrix(g)
            adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
            adj = adj + sp.eye(adj.shape[0])
            adj = normalize_adj(adj).todense()
            assert len(cascade_lst) == len(timespan_lst)
            if len(adj) > len(cascade_lst):
                continue
            assert len(adj) == len(cascade_lst), "the length of cascade not equals to the length of nodes list."
            node2time = {k:v for k,v in zip(cascade_lst, timespan_lst)}
            new_cascade_lst = list(g.nodes())
            new_timespan_lst = [node2time[k] for k in new_cascade_lst]
            # assert new_cascade_lst == list(g.nodes()), "Order of the cascade list not equal the graph node list"
            cascades_info[cascade_id] = [g, new_cascade_lst, new_timespan_lst, adj, label]
    write_train_val_test(cascades_info, file_type)

def write_train_val_test(cascades_info, file_type=0):
    # graph_lst, cas_src, time_src, adj_list, tgt = cascades_info
    train_file_path = file_path + "/train.pkl"
    valid_file_path = file_path + "/valid.pkl"
    test_file_path = file_path + "/test.pkl"
    if file_type == 0:
        hf = open(train_file_path, 'wb')
    elif file_type == 1:
        hf = open(valid_file_path, 'wb')
    else:
        hf = open(test_file_path, 'wb')

    # generate shortest path length matrix, time interval difference matrix and lca matrix
    # k = 0
    dataset = {}
    for cascade_id, cascades in cascades_info.items():
        g, cas_src, td, adj, tgt = cascades
        # spl_lst, td_lst, lca_lst = [], [], []
        # traverse all data, record the graph, the activate nodes, the graph node list and the time difference list
        act_nodes, cas_graph_nodes = cas_src, list(g.nodes())
        n = len(act_nodes)
        # init all matrix, length = padding length
        spl_matrix = np.zeros((n, n), dtype=np.int)
        td_matrix = np.zeros((n, n))
        lca_matrix = np.zeros((n, n), dtype=np.int)
        # record shortest path length and the shortest paths of root
        undir_g = g.copy().to_undirected()
        spl = dict(nx.all_pairs_shortest_path_length(undir_g))
        sp = nx.shortest_path(g, cas_graph_nodes[0])

        all_rt2tg_sp = []
        for node in list(cas_graph_nodes):
            all_rt2tg_sp.append(sp[node])

        for i in range(n):
            for j in range(i, n):
                # update u->v and v->u
                spl_matrix[i][j] = spl[cas_graph_nodes[i]][cas_graph_nodes[j]] + 1
                td_matrix[i][j] = td[j] - td[i] + 1

                spl_matrix[j][i] = spl[cas_graph_nodes[i]][cas_graph_nodes[j]] + 1
                td_matrix[j][i] = td[i] - td[j] + 1
                # Cartesian matching to solve LCA
                source = all_rt2tg_sp[i]
                target = all_rt2tg_sp[j]
                relation = [node for node in source if node in target]
                if not relation:
                    lca_matrix[i][j] = 0
                else:
                    lca_matrix[i][j] = relation[-1]
        lca_matrix += lca_matrix.T - np.diag(lca_matrix.diagonal())
        # calculate lpe
        lpe_matrix = np.zeros((n, m))
        # adj = torch.FloatTensor(adj)
        eig_values, eig_vectors = np.linalg.eigh(adj)
        indices = m if m < len(eig_values) else len(eig_values)
        # eig_values, eig_vectors = eig_values.numpy(), eig_vectors.numpy()
        # indices = np.argsort(eig_values)[:m]
        lpe_matrix[:, :indices] = np.array(eig_vectors[:, :indices])

        dataset.setdefault("cascade_id", []).append(cascade_id)
        dataset.setdefault("cascade_src", []).append(cas_src)
        dataset.setdefault("temporal_src", []).append(td)
        dataset.setdefault("spl_matrix", []).append(spl_matrix)
        dataset.setdefault("td_matrix", []).append(td_matrix)
        dataset.setdefault("lca_matrix", []).append(lca_matrix)
        dataset.setdefault("lpe_matrix", []).append(lpe_matrix)
        dataset.setdefault("labels", []).append(tgt)
    pickle.dump(dataset, hf)
    hf.close()

def main():
    start_time = time.time()
    print("generate training data......")
    load_cascade(file_path, 0)
    end_time = time.time()
    print("generate training data successful! costing time: {:.3f}".format((end_time-start_time)/60))

    start_time = time.time()
    print("generate validation data......")
    load_cascade(file_path, 1)
    end_time = time.time()
    print("generate validation data successful! costing time: {:.3f}".format((end_time-start_time)/60))

    start_time = time.time()
    print("generate testing data......")
    load_cascade(file_path, 2)
    end_time = time.time()
    print("generate testing data successful!costing time: {:.3f}".format((end_time-start_time)/60))
if __name__ == '__main__':
    main()
