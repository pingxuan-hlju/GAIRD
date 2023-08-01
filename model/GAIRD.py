import torch.nn as nn
import torch
import torch.nn.functional as F


def construct_embedding(separated_features, features, lncRNAs, Diseases):
    embedding_left = torch.cat((features[lncRNAs.tolist()], features[Diseases.tolist()]), 1)
    embedding_left = embedding_left.view(lncRNAs.shape[0], 1, 2, features.shape[1])

    embedding_right = torch.cat((separated_features[lncRNAs.tolist()], separated_features[Diseases.tolist()]), 1)
    embedding_right = embedding_right.view(lncRNAs.shape[0], 1, 2, separated_features.shape[1])

    return torch.cat((embedding_left, embedding_right), dim=3)


def mix_channels(x, groups):
    batch_size, channels, height, width = x.size()
    channels_per_group = int(channels / groups)
    res = x.view(batch_size, groups, channels_per_group, height, width)
    res = res.transpose(1, 2).contiguous()
    res = res.view(batch_size, -1, height, width)

    return res


class NRUM(nn.Module):
    def __init__(self, input_dim, attn_dim, gru_layer_num):
        super(NRUM, self).__init__()
        self.FCNN_position = nn.Linear(in_features=input_dim, out_features=input_dim, bias=True)

        self.fc_attn = nn.Linear(attn_dim, 1, bias=False)
        self.fc_transform = nn.Linear(input_dim, attn_dim, bias=True)

        self.hom_combiner = nn.Linear(input_dim + input_dim, 1, bias=False)
        self.het_combiner = nn.Linear(input_dim + input_dim, 1, bias=False)

        self.hom_encoding = nn.GRU(input_dim, input_dim, gru_layer_num, batch_first=False, bidirectional=False)
        self.het_encoding = nn.GRU(input_dim, input_dim, gru_layer_num, batch_first=False, bidirectional=False)

    def forward(self, features, hom_paths, het_paths, SPD):
        path_num = int(len(hom_paths) / features.shape[0])
        left_list = [i for i in range(features.shape[0]) for j in range(path_num)]

        # Calculate the homogeneity distribution
        hom_path_embeddings, _ = self.hom_encoding(features[hom_paths])
        hom_path_embeddings = hom_path_embeddings[:, -1, :]
        hom_embedding = torch.cat((features[left_list], hom_path_embeddings), dim=1)
        hom_score = F.leaky_relu(self.hom_combiner(hom_embedding))
        hom_score = hom_score.view(features.shape[0], path_num, 1)
        hom_score = F.softmax(hom_score, dim=1)
        hom_path_embeddings = hom_path_embeddings.view(features.shape[0], path_num, -1).permute(0, 2, 1)
        hom_distributions = torch.matmul(hom_path_embeddings, hom_score)[:, :, 0]

        # Calculate the heterogeneity distribution
        het_path_embeddings, _ = self.het_encoding(features[het_paths])
        het_path_embeddings = het_path_embeddings[:, -1, :]
        het_embedding = torch.cat((features[left_list], het_path_embeddings), dim=1)
        het_score = F.leaky_relu(self.het_combiner(het_embedding))
        het_score = het_score.view(features.shape[0], path_num, 1)
        het_score = F.softmax(het_score, dim=1)
        het_path_embeddings = het_path_embeddings.view(features.shape[0], path_num, -1).permute(0, 2, 1)
        het_distributions = torch.matmul(het_path_embeddings, het_score)[:, :, 0]

        # Calculate the position distribution
        pos_distributions = self.FCNN_position(SPD)

        # Attention mechanisms at the distribution level
        beta = []
        for distribution in [hom_distributions, het_distributions, pos_distributions]:
            tmp = torch.tanh(self.fc_transform(distribution))
            score = self.fc_attn(tmp)
            beta.append(score)
        beta = torch.stack(beta, dim=2)
        beta = F.softmax(beta, dim=2)
        distributions = torch.stack([hom_distributions, het_distributions, pos_distributions], dim=1)
        enhanced_features = torch.matmul(beta, distributions)
        enhanced_features = enhanced_features.view(enhanced_features.shape[0], -1)
        return enhanced_features


class NRDM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NRDM, self).__init__()

        self.FCNN_topology = nn.Linear(in_features=input_dim, out_features=input_dim, bias=True)
        self.FCNN_attribute = nn.Linear(in_features=input_dim * 2, out_features=input_dim, bias=True)
        self.FCNN_initial = nn.Linear(in_features=input_dim, out_features=input_dim, bias=False)

        self.W = nn.Parameter(torch.empty(size=(input_dim, output_dim)))
        self.params = nn.Parameter(torch.rand(3))

    def forward(self, enhanced_features, features, A):
        # Purify node attribute using original attribute
        pur_attributes = self.FCNN_attribute(torch.cat((enhanced_features, features), dim=1))

        # Purify network topology using representation subtraction
        p = F.softmax(enhanced_features, dim=1)
        q = F.softmax(features, dim=1)
        pur_topology = self.FCNN_topology(torch.mul(q, torch.log10(p) - torch.log10(q)))

        # independent channel for the initial attributes
        initial_attributes = self.FCNN_initial(features)

        # integrate pure attribute, pure topology and initial attribute using Adaptive-GCN
        score = F.softmax(self.params, dim=0)
        H = F.relu(torch.mm(
                torch.mm(A, pur_attributes) * score[0] + torch.mm(A, pur_topology) * score[1] +
                initial_attributes * score[2], self.W
            ))
        return H


class MFIM(nn.Module):
    def __init__(self, cnn_out_dim, dropout_rate):
        super(MFIM, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 10), padding=1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(2, 2), stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 10), padding=1)
        )
        self.depth_conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 2), stride=1, padding=1, groups=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 10))
        )
        self.depth_conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(2, 2), stride=1, padding=1, groups=8),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 10))
        )
        self.out = nn.Sequential(
            nn.Linear(608, cnn_out_dim),
            nn.Dropout(dropout_rate),
            nn.PReLU(),
            nn.Linear(cnn_out_dim, 2)
        )

    def forward(self, x):
        # first group convolution: traditional convolution
        conv_1 = self.conv_1(x)
        conv_2 = self.conv_2(x)
        conv1 = torch.cat((conv_1, conv_2), dim=1)

        # channel mixing
        conv1 = mix_channels(conv1, 2)

        # second group convolution: depth separable convolution
        conv2_1 = self.depth_conv_1(conv1[:, 0:8, :, :])
        conv2_2 = self.depth_conv_2(conv1[:, 8:16, :, :])
        conv2 = torch.cat((conv2_1, conv2_2), dim=1)

        # predicting the lncRNA-Disease Association Score
        score = self.out(conv2.view(x.shape[0], -1))

        return score


class GAIRD(nn.Module):
    def __init__(self, input_dim, output_dim, cnn_out_dim, attn_dim, gru_layer_num, dropout_rate):
        super(GAIRD, self).__init__()
        self.NRUM_layer = NRUM(input_dim, attn_dim, gru_layer_num)
        self.NRDM_layer = NRDM(input_dim, output_dim)
        self.MFIM_layer = MFIM(cnn_out_dim, dropout_rate)

    def forward(self, features, A, hom_paths, het_paths, SPD, lncRNAs, diseases):
        # updating the representation of nodes with Homogeneity, heterogeneity and position distributions
        enhanced_features = self.NRUM_layer(features, hom_paths, het_paths, SPD)

        # decoupling node attribute and network topology using purification operation and Adaptive-GCN
        decoupled_features = self.NRDM_layer(features, enhanced_features, A)

        # integrating multi-level feature from pairwise view and assessing association scores
        embeddings = construct_embedding(decoupled_features, features, lncRNAs, diseases)
        score = self.MFIM_layer(embeddings)
        return score
