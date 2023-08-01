import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.utils.data import DataLoader

from config.configs import TRAIN_TEST_SET_PATH, PREPROCESSED_DATA_PATH, CONFIG, DEVICE, MODEL_PATH, SCORE_PATH
from model.GAIRD import GAIRD
from utils.dataset import Datasets
from utils.evaluation import fold_5, curve, calculate_TPR_FPR
from utils.utils import generating_paths


def run_train():
    # Load training set and test set from file
    data = np.load(TRAIN_TEST_SET_PATH)
    train_index, label = data['train_set'], data['label']

    # Load pre-processed data
    preprocessed_data = np.load(PREPROCESSED_DATA_PATH)

    for fold in range(5):  # Five-fold cross validation
        # Extract the basic data needed for the model
        feature = torch.FloatTensor(preprocessed_data['features'][fold]).to(DEVICE)
        A = torch.FloatTensor(preprocessed_data['adjacency'][fold]).to(DEVICE)
        SPD = torch.FloatTensor(preprocessed_data['SPD'][fold]).to(DEVICE)

        # Packing the dataset, defining the model, optimiser and loss function
        train_dataset = DataLoader(Datasets(train_index[fold], label), CONFIG["batch_size"], shuffle=True)
        net = GAIRD(input_dim=CONFIG['input_dim'],
                    output_dim=CONFIG['output_dim'],
                    cnn_out_dim=CONFIG['cnn_out_dim'],
                    dropout_rate=CONFIG['dropout_rate'],
                    attn_dim=CONFIG['attn_dim'],
                    gru_layer_num=CONFIG['gru_layer_num']).to(DEVICE)
        optimizer = torch.optim.Adam(net.parameters(), lr=CONFIG["lr"])
        loss_function = nn.CrossEntropyLoss()

        net.train()  # train
        for epoch in range(CONFIG["epoch"]):
            hom_paths, het_paths = generating_paths(preprocessed_data['adjacency'][fold], CONFIG['threshold'],
                                                    CONFIG['bfs_p'], CONFIG['bfs_q'], CONFIG['dfs_p'], CONFIG['dfs_q'],
                                                    CONFIG['path_num'], CONFIG['path_length'])
            train_loss_records = []
            for step, (l, d, y) in enumerate(train_dataset):
                scores = net(feature, A, hom_paths, het_paths, SPD, l, d + label.shape[0])
                y = Variable(y).long().to(DEVICE)
                train_loss = loss_function(scores, y)
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()
                train_loss_records.append(train_loss.item())
                print(step, train_loss.item())
            train_loss = round(sum(train_loss_records) / len(train_loss_records), 4)
            print(f"[train]   Fold: {fold + 1} / {5}, Epoch: {epoch + 1} / {CONFIG['epoch']}, Loss: {train_loss}")
            torch.save(net.state_dict(), MODEL_PATH % fold)


def run_test():
    # Load training set and test set from file
    data = np.load(TRAIN_TEST_SET_PATH)
    test_index, label = data['test_set'], data['label']

    # Load pre-processed data
    preprocessed_data = np.load(PREPROCESSED_DATA_PATH)

    matrices = []
    for fold in range(5):  # Five-fold cross validation
        # Extract the basic data needed for the model
        feature = torch.FloatTensor(preprocessed_data['features'][fold]).to(DEVICE)
        A = torch.FloatTensor(preprocessed_data['adjacency'][fold]).to(DEVICE)
        SPD = torch.FloatTensor(preprocessed_data['SPD'][fold]).to(DEVICE)
        hom_paths, het_paths = generating_paths(preprocessed_data['adjacency'][fold], CONFIG['threshold'],
                                                CONFIG['bfs_p'], CONFIG['bfs_q'], CONFIG['dfs_p'], CONFIG['dfs_q'],
                                                CONFIG['path_num'], CONFIG['path_length'])

        # Packing the dataset, and loading trained model
        train_dataset = DataLoader(Datasets(test_index, label), CONFIG["batch_size"], shuffle=True)
        net = GAIRD(input_dim=CONFIG['input_dim'],
                    output_dim=CONFIG['output_dim'],
                    cnn_out_dim=CONFIG['cnn_out_dim'],
                    dropout_rate=CONFIG['dropout_rate'],
                    attn_dim=CONFIG['attn_dim'],
                    gru_layer_num=CONFIG['gru_layer_num']).to(DEVICE)
        net.load_state_dict(torch.load(MODEL_PATH % fold, map_location=torch.device("cpu")))

        net.eval()  # test
        matrix = np.full((label.shape[0], label.shape[1]), 0, dtype=float)
        for step, (l, d, y) in enumerate(train_dataset):
            print(step)
            with torch.no_grad():
                scores = net(feature, A, hom_paths, het_paths, SPD, l, d + label.shape[0])
            scores = F.softmax(scores, dim=1)
            for index in range(scores.shape[0]):
                matrix[l[index]][d[index]] = scores[index][1]
        matrices.append(matrix)

    # Save score matrices
    np.save(SCORE_PATH, np.array(matrices))


def run_evaluate():
    # Load predication score of 240 lncRNAs and 405 diseases
    scores = np.load(SCORE_PATH)

    # Load train index and label
    data = np.load(TRAIN_TEST_SET_PATH)
    train_index, label = data['train_set'], data['label']

    FPRs = []
    TPRs = []
    Ps = []
    for fold, (score, index) in enumerate(zip(scores, train_index)):
        for i in range(index.shape[0]):
            score[index[i][0]][index[i][1]] = -1
            label[index[i][0]][index[i][1]] = -1
        f = np.zeros(shape=(score.shape[0], 1))
        for i in range(score.shape[0]):
            f[i] = np.sum(score[i] > -1)
        TPR, FPR, P = calculate_TPR_FPR(score, f, label)
        FPRs.append(FPR)
        TPRs.append(TPR)
        Ps.append(P)

    TPR_5, FPR_5, PR_5 = fold_5(TPRs, FPRs, Ps)
    curve(FPR_5, TPR_5, PR_5)


if __name__ == '__main__':
    run_train()
    run_test()
    run_evaluate()
