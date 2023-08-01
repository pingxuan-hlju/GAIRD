import numpy as np
from config.configs import L_D_PATH, M_D_PATH, L_M_PATH, M_M_PATH, D_D_PATH, TRAIN_TEST_SET_PATH, \
    PREPROCESSED_DATA_PATH

from utils.utils import split_dataset, create_feature, floyd


if __name__ == '__main__':
    # Load dataset from file
    l_d = np.loadtxt(L_D_PATH, dtype=int)
    d_d = np.loadtxt(D_D_PATH, dtype=float)
    m_m = np.loadtxt(M_M_PATH, dtype=float)
    m_d = np.loadtxt(M_D_PATH, dtype=int)
    l_m = np.loadtxt(L_M_PATH, dtype=int)
    data = {'l_d': l_d, 'd_d': d_d, 'm_m': m_m, 'm_d': m_d, 'l_m': l_m}

    # Generate and save training and test sets
    train_index, test_index = split_dataset(data['l_d'])
    np.savez(TRAIN_TEST_SET_PATH, train_set=train_index, test_set=test_index, label=data['l_d'])

    # Generate and save node feature
    features = []
    for fold in range(5):
        feature = create_feature(data['l_d'], data['d_d'], data['m_d'], data['l_m'],
                                 data['m_m'], train_index[fold])
        features.append(feature)

    # Calculate the shortest path
    SPDs = []
    for fold in range(5):
        SPD = np.where(features[fold] == 0, float('inf'), features[fold])
        np.fill_diagonal(SPD, 0)
        SPD = floyd(SPD)
        SPD = np.where(SPD == float('inf'), -1, SPD)
        SPDs.append(SPD)

    # Save data
    np.savez(PREPROCESSED_DATA_PATH, features=features, adjacency=features, SPD=SPDs)
