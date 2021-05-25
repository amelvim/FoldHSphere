#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: main_optimize_prototypes.py
# Created Date: Monday, February 8th 2021, 10:43:12 am
# Author: Amelia Villegas-Morcillo
#
# Copyright (c) 2021 Amelia Villegas-Morcillo
###


import argparse
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from plot_similarity import plot_matsim_hist
from thomson_loss import THLSum, THLMaxCosine, THLMaxInverseDist


def create_parser_args():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--input_file", type=str,
                        default="initial_cnn-gru-softmax.npy")
    parser.add_argument("--loss_type", type=str, default="sum", const="sum",
                        nargs="?", choices=["sum", "max_cos", "max_invdist"])
    parser.add_argument("--init_lr", type=float, default=0.001)
    parser.add_argument("--num_epochs", type=int, default=6000)
    parser.add_argument("--print_step", type=int, default=1)
    parser.add_argument("--save_start", type=int, default=1)
    parser.add_argument("--save_end", type=int, default=6000)
    parser.add_argument("--output_dir", type=str,
                        default="optim_cnn-gru-softmax/thl_sum")
    return parser.parse_args()


def get_loss_func(loss_type):
    loss_dict = {"sum": THLSum, "max_cos": THLMaxCosine,
                 "max_invdist": THLMaxInverseDist}
    return loss_dict.get(loss_type, "nothing")


def compute_thl_sum_max(x):
    norm = torch.norm(x, p=2, dim=-1, keepdim=True)
    x_norm = torch.div(x, norm)
    dist = F.pdist(x_norm, p=2)
    return torch.sum(1 / dist), torch.max(1 / dist)


def compute_cosines(x):
    idx = torch.triu_indices(x.shape[0], x.shape[0], offset=1)
    cos_mat = F.cosine_similarity(x.unsqueeze(1), x.unsqueeze(0), dim=2)
    cosines = cos_mat[idx[0], idx[1]]
    return cos_mat.data.cpu().numpy(), cosines.data.cpu().numpy()


def main(args):
    # Set log file and Tensorboard writer
    log_file = args.output_dir + "/logs.txt"
    writer = SummaryWriter(log_dir=args.output_dir)

    # Load initial prototypes, L2-normalize and convert to Tensor
    prototypes = np.load(args.input_file)
    prototypes /= np.linalg.norm(prototypes, ord=2, axis=1, keepdims=True)
    prototypes = torch.tensor(prototypes)

    # Plot initial cosine similarity histogram
    cos_mat, cosines = compute_cosines(prototypes)
    fig = plot_matsim_hist(cos_mat, cosines)
    writer.add_figure("figure", fig, global_step=0)

    # Move prototypes matrix to cuda and init autograd
    prototypes = prototypes.cuda()
    prototypes.requires_grad = True

    # Initialize loss function and optimizer
    criterion = get_loss_func(args.loss_type)().cuda()
    optimizer = torch.optim.Adam([prototypes], lr=args.init_lr)

    # Optimization epochs
    for ep in range(args.num_epochs):
        optimizer.zero_grad()
        loss = criterion(prototypes)
        loss.backward()
        optimizer.step()

        if ((ep+1) % args.print_step == 0):
            # Compute metrics
            loss_sum, loss_max = compute_thl_sum_max(prototypes)
            cos_mat, cosines = compute_cosines(prototypes)
            cosine_max = np.max(cosines)

            # Print metrics to file
            with open(log_file, "a") as f:
                print(
                    "Epoch [%d, %d]: loss_sum=%f, loss_max=%f, cosine_max=%f" \
                    % (ep+1, args.num_epochs, loss_sum.item(), loss_max.item(), cosine_max),
                    file=f
                )
                print(torch.norm(prototypes.data.cpu(), p=2, dim=-1), file=f)

            # Write metrics to Tensorboard
            writer.add_scalar("loss_sum_epoch", loss_sum, global_step=ep+1)
            writer.add_scalar("loss_max_epoch", loss_max, global_step=ep+1)
            writer.add_scalar("cosine_max_epoch", cosine_max, global_step=ep+1)

            # Write cosine similarity histogram plot to Tensorboard
            fig = plot_matsim_hist(cos_mat, cosines)
            writer.add_figure("figure", fig, global_step=ep+1)

            if ((ep+1) >= args.save_start) and ((ep+1) <= args.save_end):
                # Save optimized prototypes matrix to file
                np.save(args.output_dir + "/prototypes_ep%d.npy" % (ep+1),
                        prototypes.data.cpu().numpy())

    # Close Tensorboard writer
    writer.close()


if __name__ == "__main__":
    args = create_parser_args()
    main(args)
