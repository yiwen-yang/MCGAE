import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
from sklearn.decomposition import PCA


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCN(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout=0.0, use_pca=False):
        super(GCN, self).__init__()
        self.n_hidden = n_hidden
        self.dropout = dropout
        self.use_pca = use_pca
        self.gc1 = GraphConvolution(n_input, n_hidden)
        self.gc2 = GraphConvolution(n_hidden, n_output)

    def apply_pca(self,x):
        # Note: Using GPU tensors in linear algebra operations (like PCA) might require special handling or libraries
        x_cpu = x.cpu().detach().numpy()  # Move data to CPU and convert to NumPy array
        pca = PCA(n_components=self.n_hidden)
        x_pca = pca.fit_transform(x_cpu)
        return torch.tensor(x_pca, device=x.device, dtype=x.dtype)  # Move data back to the original device

    def forward(self, x, adj):
        if self.use_pca:
            x = self.apply_pca(x)
        else:
            x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout)
        x = self.gc2(x, adj)
        return x


class Attention(nn.Module):
    def __init__(self, n_input, n_hidden=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh(),
            nn.Linear(n_hidden, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta


class MLPDecoder(nn.Module):
    def __init__(self, n_input, n_hidden, n_output, dropout=0.0):
        super(MLPDecoder, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)  # Add dropout layer
        self.fc2 = nn.Linear(n_hidden, n_output)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Add dropout layer
        x = self.fc2(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(n_h, n_h, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()
    @staticmethod
    def forward(emb, mask=None):  # def forward(self, emb, mask=None):
        if mask is None:
            mask = torch.eye(emb.size(0), dtype=emb.dtype, device=emb.device)
        vsum = torch.mm(mask, emb)
        row_sum = torch.sum(mask, 1)
        row_sum = row_sum.view(-1, 1).expand(-1, vsum.size(1))
        global_emb = vsum / row_sum

        return F.normalize(global_emb, p=2, dim=1)


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )

        # 均值和方差线性层
        self.fc_mean = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid(),
        )

    def encode(self, x):
        hidden = self.encoder(x)
        mean = self.fc_mean(hidden)
        logvar = self.fc_logvar(hidden)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z

    def decode(self, z):
        reconstructed = self.decoder(z)
        return reconstructed

    def forward(self, x):
        # 编码器
        mean, logvar = self.encode(x)
        # 重参数化
        z = self.reparameterize(mean, logvar)
        # 解码器
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar

    def loss_function_vae(self, reconstructed, inputs, mean, logvar):
        # 计算重构损失（reconstruction loss）
        reconstruction_loss = F.mse_loss(reconstructed, inputs, reduction="mean")

        # 计算KL散度（KL divergence）
        kl_divergence = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())

        # 综合重构损失和KL散度作为总损失
        total_loss = reconstruction_loss + kl_divergence * 0.01

        return total_loss

    def train_vae(self, optimizer, train_data, num_epochs, batch_size):
        self.train()  # 设置模型为训练模式

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            # 随机打乱训练数据
            indices = torch.randperm(len(train_data))
            train_data = train_data[indices]
            for i in range(0, len(train_data), batch_size):
                batch = train_data[i : i + batch_size]
                inputs = batch.view(-1, self.input_dim)

                optimizer.zero_grad()  # 梯度清零

                reconstructed, mean, logvar = self(inputs)  # 前向传播
                loss = self.loss_function_vae(
                    reconstructed, inputs, mean, logvar
                )  # 计算损失
                epoch_loss += loss.item()

                loss.backward()  # 反向传播
                optimizer.step()  # 更新模型参数
            if epoch % 10 == 0:
                print(
                    "For denoise data training:",
                    "Epoch [{}/{}], Loss: {:.4f}".format(
                        epoch, num_epochs, epoch_loss / len(train_data)
                    ),
                )
