#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
author@tao
"""

import train_cls
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class Transformer_Layer(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.q_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.k_conv = nn.Conv1d(channels, channels // 4, 1, bias=False)
        self.q_conv.weight = self.k_conv.weight
        self.v_conv = nn.Conv1d(channels, channels, 1)
        self.trans_conv = nn.Conv1d(channels, channels, 1)
        self.after_norm = nn.BatchNorm1d(channels)
        self.act = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x_q = self.q_conv(x).permute(0, 2, 1)  # b, n, c
        x_k = self.k_conv(x)  # b, c, n
        x_v = self.v_conv(x)
        energy = x_q @ x_k  # b, n, n
        attention = self.softmax(energy)
        attention = attention / (1e-9 + attention.sum(dim=1, keepdims=True))
        # attention = attention / attention.sum(dim=1, keepdim=True)

        x_r = x_v @ attention  # b, c, n
        x_r = self.act(self.after_norm(self.trans_conv(x - x_r)))
        x = x + x_r
        return x


def random_select(idx_num, layer_num, g_k, batch_size):
    index = torch.stack([torch.randperm(idx_num)[:layer_num] for _ in range(batch_size * g_k)])
    index = index.view(batch_size, g_k * layer_num, -1).cuda()
    return index


class get_global(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.trans1 = Transformer_Layer(64)
        self.trans2 = Transformer_Layer(128)
        self.trans3 = Transformer_Layer(256)
        self.trans4 = Transformer_Layer(512)

        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(256, 512, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        g_k = int(self.args.g_k)
        b, c, n = x.shape
        x = x.transpose(2, 1)
        idx = random_select(n, 512, g_k, b)
        x1 = torch.gather(x, 1, idx.expand(-1, -1, x.size(-1)))
        x1 = x1.view(b, g_k, 512, -1).permute(0, 3, 1, 2)
        x1 = self.conv1(x1)
        x1 = x1.max(dim=2, keepdim=False)[0]
        x1 = self.trans1(x1)

        b, c, n = x1.shape
        x1 = x1.transpose(2, 1)
        idx = random_select(n, 256, g_k, b)
        x2 = torch.gather(x1, 1, idx.expand(-1, -1, x1.size(-1)))
        x2 = x2.view(b, g_k, 256, -1).permute(0, 3, 1, 2)
        x2 = self.conv2(x2)
        x2 = x2.max(dim=2, keepdim=False)[0]
        x2 = self.trans2(x2)

        b, c, n = x2.shape
        x2 = x2.transpose(2, 1)
        idx = random_select(n, 128, g_k, b)
        x3 = torch.gather(x2, 1, idx.expand(-1, -1, x2.size(-1)))
        x3 = x3.view(b, g_k, 128, -1).permute(0, 3, 1, 2)
        x3 = self.conv3(x3)
        x3 = x3.max(dim=2, keepdim=False)[0]
        x3 = self.trans3(x3)

        b, c, n = x3.shape
        x3 = x3.transpose(2, 1)
        idx = random_select(n, 64, g_k, b)
        x4 = torch.gather(x3, 1, idx.expand(-1, -1, x3.size(-1)))
        x4 = x4.view(b, g_k, 64, -1).permute(0, 3, 1, 2)
        x4 = self.conv4(x4)
        x4 = x4.max(dim=2, keepdim=False)[0]
        x4 = self.trans4(x4)
        x = self.conv5(x4)
        x1 = F.adaptive_max_pool1d(x, 1).view(b, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(b, -1)
        x = torch.cat((x1, x2), 1)
        return x


class get_local(nn.Module):
    def __init__(self, args):
        super(get_local, self).__init__()
        self.args = args
        self.k = args.k

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.trans1 = Transformer_Layer(64)
        self.trans2 = Transformer_Layer(64)
        self.trans3 = Transformer_Layer(128)
        self.trans4 = Transformer_Layer(256)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]
        x1 = self.trans1(x1)

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]
        x2 = self.trans2(x2)

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]
        x3 = self.trans3(x3)

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]
        x4 = self.trans4(x4)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_c = x

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        return x, x_c


class get_model(nn.Module):
    def __init__(self, args, output_channels=40):
        super().__init__()

        self.global_feature = get_global(args)
        self.local_feature = get_local(args)

        self.linear1 = nn.Linear(args.emb_dims * 4, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        gf = self.global_feature(x)
        lf, _ = self.local_feature(x)
        x = torch.cat([gf, lf], 1)
        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.linear3(x)
        return x


class get_model_part(nn.Module):
    def __init__(self, args, part_num=50):
        super().__init__()

        self.global_feature = get_global(args)
        self.local_feature = get_local(args)

        self.conv6 = nn.Conv1d(2576, 512, 1)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.conv7 = nn.Conv1d(512, 256, 1)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.conv8 = nn.Conv1d(256, part_num, 1)

    def forward(self, x, lable):
        b, c, n = x.shape
        lable = lable.transpose(2, 1).repeat(1, 1, n)
        gf = self.global_feature(x).unsqueeze(-1)
        expand = gf.repeat(1, 1, n)
        _, x_c = self.local_feature(x)
        x = torch.cat([expand, x_c, lable], 1)
        x = F.leaky_relu(self.bn6(self.conv6(x)), negative_slope=0.2)
        x = self.dp1(x)
        x = F.leaky_relu(self.bn7(self.conv7(x)), negative_slope=0.2)
        x = self.dp2(x)
        x = self.conv8(x).transpose(2, 1)
        return x


def get_loss(pred, gold, smoothing=True):
    ''' Calculate cross entropy loss, apply label smoothing if needed. '''

    gold = gold.contiguous()

    if smoothing:
        eps = 0.2
        n_class = pred.size(1)

        one_hot = torch.zeros_like(pred).scatter(1, gold.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = F.log_softmax(pred, dim=1)

        loss = -(one_hot * log_prb).sum(dim=1).mean()
    else:
        loss = F.cross_entropy(pred, gold, reduction='mean')
    return loss


if __name__ == '__main__':
    data = torch.rand((10, 3, 1024)).cuda()
    args = train_cls.parse_args()
    # model = get_global(args).cuda()
    # model = get_local(args).cuda()
    model = get_model(args).cuda()
    x = model(data)
    print(x)
    print(x.shape)
