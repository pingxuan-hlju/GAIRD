import numpy as np
import networkx
import random


def split_dataset(Interaction):
    positive_sample = np.argwhere(Interaction == 1)
    np.random.shuffle(positive_sample)
    negative_sample = np.argwhere(Interaction == 0)
    np.random.shuffle(negative_sample)

    sum_fold = int(positive_sample.shape[0] / 5)
    train_nodes = []
    for i in range(5):
        train_nodes.append(np.vstack((positive_sample[i * sum_fold:(i + 1) * sum_fold],
                                      negative_sample[i * sum_fold:(i + 1) * sum_fold])))
    train_nodes = np.array(train_nodes)

    test_index = []
    for i in range(Interaction.shape[0]):
        for j in range(Interaction.shape[1]):
            test_index.append([i, j])
    test_index = np.array(test_index)

    train_index = []
    for i in range(5):
        a = [j for j in range(5) if j != i]
        train_index.append(np.vstack((train_nodes[a])))
    train_index = np.array(train_index)

    return train_index, test_index


def create_feature(lnc_dis, dis_sim, mi_dis, lnc_mi, mi_sim, train_index):
    copy_lnc_dis = np.zeros(shape=(lnc_dis.shape[0], lnc_dis.shape[1]), dtype=int)
    for i in range(train_index.shape[0]):
        if lnc_dis[train_index[i][0]][train_index[i][1]] == 1:
            copy_lnc_dis[train_index[i][0]][train_index[i][1]] = 1

    lnc_sim = calculate_sim(lnc_dis, dis_sim)

    row_1 = np.concatenate((lnc_sim, copy_lnc_dis, lnc_mi), axis=1)
    row_2 = np.concatenate((copy_lnc_dis.T, dis_sim, mi_dis.T), axis=1)
    row_3 = np.concatenate((lnc_mi.T, mi_dis, mi_sim), axis=1)
    features = np.vstack((row_1, row_2, row_3))

    return features


def generating_paths(A, threshold, bfs_p, bfs_q, dfs_p, dfs_q, path_num, path_length):
    B = np.where(A > threshold, 1, 0)
    np.fill_diagonal(B, 0)
    G = networkx.from_numpy_array(B)

    hom_paths = []
    het_paths = []
    for i in range(A.shape[0]):
        hom_paths.extend(sample_paths(G, i, path_num, path_length, bfs_p, bfs_q))
        het_paths.extend(sample_paths(G, i, path_num, path_length, dfs_p, dfs_q))
    return hom_paths, het_paths


def calculate_sim(Interaction, original_sim):
    target_sim = np.zeros(shape=(Interaction.shape[0], Interaction.shape[0]), dtype=float)
    for i in range(target_sim.shape[0]):
        for j in range(target_sim.shape[1]):
            if i == j:
                target_sim[i][j] = 1
            else:
                l1_num = np.sum(Interaction[i] == 1.0)
                l2_num = np.sum(Interaction[j] == 1.0)
                if l1_num == 0 or l2_num == 0:
                    target_sim[i][j] = 0
                else:
                    l1_index = np.where(Interaction[i] == 1.0)
                    l2_index = np.where(Interaction[j] == 1.0)
                    sim_sum = 0.0
                    for l in range(len(l1_index[0])):
                        sim_sum = sim_sum + np.max(original_sim[l1_index[0][l]][l2_index[0]])
                    for l in range(len(l2_index[0])):
                        sim_sum = sim_sum + np.max(original_sim[l2_index[0][l]][l1_index[0]])
                    target_sim[i][j] = sim_sum / (l1_num + l2_num)
    return target_sim


def sample_paths(G, start, path_num, path_length, p, q):
    path_list = []
    for i in range(path_num):
        path = [start]
        while len(path) < path_length:
            cur = path[-1]
            neighbors = list(G.neighbors(cur))
            if len(neighbors) == 0:
                path.append(cur)
                continue
            if len(neighbors) == 1:
                path.append(neighbors[0])
                continue
            prev = path[-2] if len(path) > 1 else -1
            weights = []
            for next in neighbors:
                if next == prev:
                    weights.append(1 / p)
                elif G.has_edge(next, prev):
                    weights.append(1)
                else:
                    weights.append(1 / q)
            next = random.choices(neighbors, weights)[0]
            path.append(next)
        path_list.append(path)
    return path_list


def floyd(A):
    for k in range(len(A)):
        for i in range(len(A)):
            for j in range(len(A)):
                if A[i][k] + A[k][j] < A[i][j]:
                    A[i][j] = A[i][k] + A[k][j]
    return A
