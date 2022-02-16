import argparse
import math
import random
import os
import copy

import numpy as np
import torch
torch.backends.cudnn.benchmark = True
from torch import nn, autograd, optim
from torch.nn import functional as F
from torch.utils import data
import torch.distributed as dist
from torchvision import transforms, utils
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import lpips
from model_ae import Discriminator, Denoiser
import id_loss


try:
    import wandb

except ImportError:
    wandb = None


from dataset import FaceDataset
from distributed import (
    get_rank,
    synchronize,
    reduce_loss_dict,
    reduce_sum,
    get_world_size,
)
from op import conv2d_gradfix


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)

    if shuffle:
        return data.RandomSampler(dataset)

    else:
        return data.SequentialSampler(dataset)


def accumulate(model1, model2, decay=0.999):
    par1 = dict(model1.named_parameters())
    par2 = dict(model2.named_parameters())

    for k in par1.keys():
        par1[k].data.mul_(decay).add_(par2[k].data, alpha=1 - decay)


def sample_data(loader):
    while True:
        for batch in loader:
            yield batch


def d_logistic_loss(real_pred, fake_pred):
    real_loss = F.softplus(-real_pred)
    fake_loss = F.softplus(fake_pred)

    return real_loss.mean() + fake_loss.mean()


def d_r1_loss(real_pred, real_img):
    with conv2d_gradfix.no_weight_gradients():
        grad_real, = autograd.grad(
            outputs=real_pred.sum(), inputs=real_img, create_graph=True
        )
    grad_penalty = grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()

    return grad_penalty


def g_nonsaturating_loss(fake_pred):
    loss = F.softplus(-fake_pred).mean()

    return loss


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(description="StyleGAN2 trainer")

    parser.add_argument("--path", type=str, default='/shared/rsaas/common/ffhq/images1024x1024/', help="path to the lmdb dataset")
    parser.add_argument('--arch', type=str, default='stylegan2', help='model architectures (stylegan2 | swagan)')
    parser.add_argument( "--iter", type=int, default=800000, help="total training iterations")
    parser.add_argument( "--batch", type=int, default=4, help="batch sizes for each gpus")
    parser.add_argument(
        "--n_sample",
        type=int,
        default=4,
        help="number of the samples generated during training",
    )
    parser.add_argument( "--size", type=int, default=1024, help="image sizes for the model")
    parser.add_argument( "--r1", type=float, default=10, help="weight of the r1 regularization")
    parser.add_argument(
        "--path_regularize",
        type=float,
        default=2,
        help="weight of the path length regularization",
    )
    parser.add_argument(
        "--path_batch_shrink",
        type=int,
        default=2,
        help="batch size reducing factor for the path length regularization (reduce memory consumption)",
    )
    parser.add_argument(
        "--d_reg_every",
        type=int,
        default=16,
        help="interval of the applying r1 regularization",
    )
    parser.add_argument(
        "--g_reg_every",
        type=int,
        default=4,
        help="interval of the applying path length regularization",
    )
    parser.add_argument(
        "--mixing", type=float, default=0.9, help="probability of latent code mixing"
    )
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--lr", type=float, default=0.002, help="learning rate")
    parser.add_argument( "--channel_multiplier", type=int, default=2, help="channel multiplier factor for the model. config-f = 2, else = 1",
    )
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    args = parser.parse_args()

    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = n_gpu > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    args.latent = 512
    args.n_mlp = 8

    args.start_iter = 0


    generator = Denoiser( args.size, args.latent, args.n_mlp, channel_multiplier=args.channel_multiplier).to(device).train()
    discriminator = Discriminator( args.size, channel_multiplier=args.channel_multiplier).to(device).train()
    id_encoder = id_loss.IDLoss().to(device).eval()

    enc_params = list(generator.encoder.parameters())
    dec_params = []
    for name, param in generator.decoder.named_parameters():
        #param.requires_grad = False
        if 'fuser' in name:
            enc_params.append(param)
        else:
            dec_params.append(param)

    g_optim = optim.Adam([{'params': enc_params}, {'params': dec_params, 'lr': args.lr/5}], lr=args.lr, betas=(0., 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0., 0.99)) 

    g_sched = optim.lr_scheduler.StepLR(g_optim, 300_000, gamma=0.1)
    d_sched = optim.lr_scheduler.StepLR(d_optim, 300_000, gamma=0.1)
    
    decoder_ckpt = torch.load("../SOAT/stylegan2-ffhq-config-f.pt", map_location=lambda storage, loc: storage)
    generator.decoder.load_state_dict(decoder_ckpt["g_ema"], strict=False)
    discriminator.load_state_dict(decoder_ckpt["d"])

    if args.ckpt is not None:
        print("load model:", args.ckpt)

        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)

        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_optim.load_state_dict(ckpt['g_optim'])
        d_optim.load_state_dict(ckpt['d_optim'])
        g_sched.load_state_dict(ckpt['g_sched'])
        d_sched.load_state_dict(ckpt['d_sched'])
        args.start_iter = ckpt['iter']

    if args.distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
        )

    transform = transforms.Compose(
        [
            transforms.Resize((args.size, args.size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    #lpips_criterion = lpips.LPIPS(net='vgg').to(device)

    dataset = FaceDataset(args.size)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=args.distributed),
        drop_last=True,
    )
    writer_path = './runs_1024'
    os.makedirs(writer_path, exist_ok=True)
    os.makedirs('checkpoint_1024', exist_ok=True)
    writer = SummaryWriter(writer_path)

    loader = sample_data(loader)

    pbar = range(args.iter)

    if get_rank() == 0:
        pbar = tqdm(pbar, initial=args.start_iter, dynamic_ncols=True)

    mean_path_length = 0

    d_loss_val = 0
    r1_loss = torch.tensor(0.0, device=device)
    g_loss_val = 0
    path_loss = torch.tensor(0.0, device=device)
    path_lengths = torch.tensor(0.0, device=device)
    mean_path_length_avg = 0
    loss_dict = {}

    if args.distributed:
        g_module = generator.module
        d_module = discriminator.module

    else:
        g_module = generator
        d_module = discriminator

    accum = 0.5 ** (32 / (10 * 1000))
    sample_z = torch.randn(args.n_sample, args.latent, device=device)

    for idx in pbar:
        i = idx + args.start_iter

        if i > args.iter:
            print("Done!")

            break

        im_lq, im_hq  = next(loader)
        im_lq = im_lq.to(device)
        im_hq = im_hq.to(device)

        with torch.no_grad():
            target_id = id_encoder.encode(im_hq.mean(1, keepdim=True).repeat(1,3,1,1))

        fake_img = generator(im_lq, target_id)

        real_pred, _ = discriminator(im_hq)
        fake_pred, _ = discriminator(fake_img.detach())
        d_loss = d_logistic_loss(real_pred, fake_pred)

        writer.add_scalar('d_loss', d_loss, i)

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        d_regularize = i % args.d_reg_every == 0

        if d_regularize:
            im_hq.requires_grad = True

            real_pred, _ = discriminator(im_hq)
            r1_loss = d_r1_loss(real_pred, im_hq)

            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        
        fake_pred, fake_feats = discriminator(fake_img)
        with torch.no_grad():
            _, real_feats = discriminator(im_hq)

        g_loss = .1*g_nonsaturating_loss(fake_pred)
        vgg_loss = 10*sum([F.mse_loss(a, b) for a, b in zip(fake_feats, real_feats)])/len(fake_feats)
        l1_loss = 10*F.mse_loss(fake_img, im_hq)
        #vgg_loss = 10 * lpips_criterion(fake_img, im_hq).mean()

        # id loss
        fake_id = id_encoder.encode(fake_img.mean(1, keepdim=True).repeat(1,3,1,1)) #[B, 512]
        id_loss = 1*(1-(fake_id * target_id).sum(-1)).mean()

        total_loss = g_loss + vgg_loss + id_loss + l1_loss

        g_optim.zero_grad()
        total_loss.backward()
        g_optim.step()
        g_sched.step()
        d_sched.step()
        writer.add_scalar('g_loss', g_loss, i)
        writer.add_scalar('vgg_loss', vgg_loss, i)
        writer.add_scalar('id_loss', id_loss, i)
        writer.add_scalar('mse_loss', l1_loss, i)


        if get_rank() == 0:
            if i % 500 == 0:
                with torch.no_grad():
                    style, _ = g_module.encode(im_lq, target_id)
                    w_im, _ = g_module.decoder(style, input_is_latent=True)
                    id_im = generator(im_lq, torch.cat([target_id[1:], target_id[[0]]], 0))

                    sample = torch.cat([im_hq, im_lq, fake_img, id_im, w_im], 0)
                    sample = utils.make_grid(sample, normalize=True, range=(-1, 1), nrow=im_hq.size(0))
                    writer.add_image(f'{i}', sample, i)


            if (i+1) % 1000 == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "g_sched": g_sched.state_dict(),
                        "d_sched": d_sched.state_dict(),
                        "args": args,
                        "iter": i+1,
                    },
                    f"checkpoint_1024/latest.pt",
                )
