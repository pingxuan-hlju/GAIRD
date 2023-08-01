import torch
import os


def makedir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


# root
ROOT_DIR = os.path.abspath(".")

# data
DATA_DIR = os.path.join(ROOT_DIR, "data")
L_D_PATH = os.path.join(DATA_DIR, "lncRNA_disease_association.txt")
M_D_PATH = os.path.join(DATA_DIR, "miRNA_disease_association.txt")
L_M_PATH = os.path.join(DATA_DIR, "lncRNA_miRNA_interaction.txt")
M_M_PATH = os.path.join(DATA_DIR, "miRNA_similarity.txt")
D_D_PATH = os.path.join(DATA_DIR, "disease_similarity.txt")
D_NAME_PATH = os.path.join(DATA_DIR, "disease_name.txt")
L_NAME_PATH = os.path.join(DATA_DIR, "lncRNA_name.txt")

# output
OUTPUT_DIR = os.path.join(ROOT_DIR, "output")
TRAIN_TEST_SET_PATH = os.path.join(OUTPUT_DIR, "train_test_set.npz")
PREPROCESSED_DATA_PATH = os.path.join(OUTPUT_DIR, "preprocessed_data.npz")
MODEL_PATH = os.path.join(OUTPUT_DIR, "GAIRD_%d.pkl")
SCORE_PATH = os.path.join(OUTPUT_DIR, "score.npy")
makedir(OUTPUT_DIR)

# device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model configs
CONFIG = {
    "batch_size": 64,
    "lr": 0.0003,
    "epoch": 80,
    "threshold": 0.2,
    "bfs_p": 5,
    "bfs_q": 0.2,
    "dfs_p": 0.2,
    "dfs_q": 5,
    "path_num": 4,
    "path_length": 4,
    "input_dim": 1140,
    "output_dim": 800,
    "cnn_out_dim": 100,
    "dropout_rate": 0.2,
    "attn_dim": 100,
    "gru_layer_num": 2,
}
