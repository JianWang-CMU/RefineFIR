import argparse
import os
import random

import lpips
import numpy as np
import torch
from torch import autograd, nn, optim
from torch.nn import functional as F
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
from tqdm import tqdm

import id_loss
from model_ae import Denoiser, Discriminator
from op import conv2d_gradfix
from refinefir.degradation import degrade_tensor
from training_dataset import FacePairDataset


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"yes", "true", "t", "1"}:
        return True
    if value in {"no", "false", "f", "0"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def data_sampler(dataset, shuffle, distributed):
    if distributed:
        return data.distributed.DistributedSampler(dataset, shuffle=shuffle)
    return data.RandomSampler(dataset) if shuffle else data.SequentialSampler(dataset)


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
        grad_real, = autograd.grad(outputs=real_pred.sum(), inputs=real_img, create_graph=True)
    return grad_real.pow(2).reshape(grad_real.shape[0], -1).sum(1).mean()


def g_nonsaturating_loss(fake_pred):
    return F.softplus(-fake_pred).mean()


def build_parser():
    parser = argparse.ArgumentParser(description="RefineFIR trainer")
    parser.add_argument("--image-root", required=True, help="Root of clean training images, e.g. CelebHQRefForRelease.")
    parser.add_argument("--pair-list", required=True, help="Pickle file containing (source_id, target_id) pairs.")
    parser.add_argument("--warp-root", required=True, help="Directory containing precomputed warped reference images.")
    parser.add_argument("--stylegan-weight", required=True, help="StyleGAN2 decoder/discriminator checkpoint.")
    parser.add_argument("--id-weight", required=True, help="ArcFace IR-SE50 identity encoder checkpoint.")
    parser.add_argument("--ckpt", help="Optional RefineFIR checkpoint to resume from.")
    parser.add_argument("--output", default="checkpoint_refinefir", help="Checkpoint/log output directory.")
    parser.add_argument("--iter", type=int, default=800000, help="Total training iterations.")
    parser.add_argument("--batch", type=int, default=4, help="Batch size per process.")
    parser.add_argument("--n_sample", type=int, default=4)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--r1", type=float, default=10)
    parser.add_argument("--d_reg_every", type=int, default=16)
    parser.add_argument("--nref", type=int, default=1)
    parser.add_argument("--postunet", type=str2bool, default=False)
    parser.add_argument("--refloss", type=str2bool, default=True)
    parser.add_argument("--l1loss", type=float, default=0.0)
    parser.add_argument("--idloss", type=float, default=1.0)
    parser.add_argument("--cycleloss", type=float, default=20.0)
    parser.add_argument("--lr", type=float, default=0.002)
    parser.add_argument("--channel_multiplier", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--sample_every", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=0)
    return parser


def main():
    args = build_parser().parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    device = "cuda"
    n_gpu = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = n_gpu > 1

    if distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")

    rank = torch.distributed.get_rank() if distributed else 0

    generator = Denoiser(
        args.size,
        512,
        8,
        channel_multiplier=args.channel_multiplier,
        in_c=3 + 3 * args.nref,
        postunet=args.postunet,
    ).to(device).train()
    discriminator = Discriminator(args.size, channel_multiplier=args.channel_multiplier).to(device).train()
    id_encoder = id_loss.IDLoss(path=args.id_weight).to(device).eval()

    enc_params = list(generator.encoder.parameters())
    dec_params = []
    for name, param in generator.decoder.named_parameters():
        if "fuser" in name:
            enc_params.append(param)
        else:
            dec_params.append(param)
    if args.postunet:
        enc_params = list(generator.gfpunet.parameters())
        dec_params = []

    g_optim = optim.Adam([{"params": enc_params}, {"params": dec_params, "lr": args.lr / 5}], lr=args.lr, betas=(0.5, 0.99))
    d_optim = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.99))
    g_sched = optim.lr_scheduler.StepLR(g_optim, 300000, gamma=0.1)
    d_sched = optim.lr_scheduler.StepLR(d_optim, 300000, gamma=0.1)

    decoder_ckpt = torch.load(args.stylegan_weight, map_location=lambda storage, loc: storage)
    discriminator.load_state_dict(decoder_ckpt["d"])
    if args.postunet:
        generator.encoder.eval()
        generator.decoder.eval()
    else:
        generator.decoder.load_state_dict(decoder_ckpt["g_ema"], strict=False)

    start_iter = 0
    if args.ckpt:
        print("load model:", args.ckpt)
        ckpt = torch.load(args.ckpt, map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g"])
        discriminator.load_state_dict(ckpt["d"])
        g_optim.load_state_dict(ckpt["g_optim"])
        d_optim.load_state_dict(ckpt["d_optim"])
        g_sched.load_state_dict(ckpt["g_sched"])
        d_sched.load_state_dict(ckpt["d_sched"])
        start_iter = ckpt["iter"]

    if distributed:
        generator = nn.parallel.DistributedDataParallel(
            generator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
        discriminator = nn.parallel.DistributedDataParallel(
            discriminator,
            device_ids=[args.local_rank],
            output_device=args.local_rank,
            broadcast_buffers=False,
            find_unused_parameters=True,
        )

    lpips_criterion = lpips.LPIPS(net="vgg", spatial=True).to(device)
    dataset = FacePairDataset(args.image_root, args.pair_list, args.warp_root, resolution=args.size, nref=args.nref)
    loader = data.DataLoader(
        dataset,
        batch_size=args.batch,
        sampler=data_sampler(dataset, shuffle=True, distributed=distributed),
        num_workers=args.num_workers,
        drop_last=True,
    )
    loader = sample_data(loader)

    writer = None
    if rank == 0:
        os.makedirs(args.output, exist_ok=True)
        writer = SummaryWriter(os.path.join(args.output, "runs"))

    pbar = range(args.iter)
    if rank == 0:
        pbar = tqdm(pbar, initial=start_iter, dynamic_ncols=True)

    g_module = generator.module if distributed else generator
    d_module = discriminator.module if distributed else discriminator

    for idx in pbar:
        i = idx + start_iter
        if i > args.iter:
            print("Done!")
            break

        im_lq, im_hq, im_source, im_warp, degrade_params = next(loader)
        im_lq = im_lq.to(device)
        im_hq = im_hq.to(device)
        im_source = im_source.to(device)
        im_warp = im_warp.to(device)

        with torch.no_grad():
            target_id = id_encoder.encode(im_source.mean(1, keepdim=True).repeat(1, 3, 1, 1))

        fake_img = generator(torch.cat([im_lq, im_warp], 1), target_id)

        real_pred, _ = discriminator(im_hq)
        fake_pred, _ = discriminator(fake_img.detach())
        d_loss = d_logistic_loss(real_pred, fake_pred)
        if rank == 0:
            writer.add_scalar("d_loss", d_loss.item(), i)

        d_optim.zero_grad()
        d_loss.backward()
        d_optim.step()

        if i % args.d_reg_every == 0:
            im_hq.requires_grad = True
            real_pred, _ = discriminator(im_hq)
            r1_loss = d_r1_loss(real_pred, im_hq)
            discriminator.zero_grad()
            (args.r1 / 2 * r1_loss * args.d_reg_every + 0 * real_pred[0]).backward()
            d_optim.step()

        fake_pred, _ = discriminator(fake_img)
        g_loss = g_nonsaturating_loss(fake_pred)

        ds_fake_img = F.interpolate(fake_img, size=(256, 256), mode="area")
        vgg_gt_loss = lpips_criterion(ds_fake_img, F.interpolate(im_hq, size=(256, 256), mode="area"))
        if args.refloss:
            vgg_ref_loss = lpips_criterion(ds_fake_img, F.interpolate(im_warp[:, :3], size=(256, 256), mode="area"))
            vgg_loss = 10 * torch.minimum(vgg_gt_loss, vgg_ref_loss).mean()
        else:
            vgg_loss = 10 * vgg_gt_loss.mean()

        if args.l1loss:
            g_loss = g_loss + args.l1loss * torch.mean((fake_img - im_hq).abs())

        fake_img_lq = []
        for p in range(fake_img.size(0)):
            degrade_param = [a[p] for a in degrade_params]
            fake_img_lq.append(degrade_tensor(fake_img[p], *degrade_param))
        fake_img_lq = torch.stack(fake_img_lq, 0)
        cycle_loss = args.cycleloss * lpips_criterion(fake_img_lq, im_lq).mean()

        fake_id = id_encoder.encode(fake_img.mean(1, keepdim=True).repeat(1, 3, 1, 1))
        id_loss = args.idloss * (1 - (fake_id * target_id).sum(-1)).mean()

        total_loss = g_loss + vgg_loss + id_loss + cycle_loss
        g_optim.zero_grad()
        total_loss.backward()
        g_optim.step()
        g_sched.step()
        d_sched.step()

        if rank == 0:
            writer.add_scalar("g_loss", g_loss.item(), i)
            writer.add_scalar("vgg_loss", vgg_loss.item(), i)
            writer.add_scalar("id_loss", id_loss.item(), i)
            writer.add_scalar("cycle_loss", cycle_loss.item(), i)

            if i % args.sample_every == 0:
                with torch.no_grad():
                    swap_id_im = generator(torch.cat([im_lq, im_warp], 1), torch.cat([target_id[1:], target_id[[0]]], 0))
                    swap_warp_im = generator(torch.cat([im_lq, torch.cat([im_warp[1:], im_warp[[0]]], 0)], 1), target_id)
                    sample = torch.cat([im_source, im_warp[:, :3], im_hq, im_lq, fake_img, fake_img_lq, swap_id_im, swap_warp_im], 0)
                    sample = utils.make_grid(sample, normalize=True, value_range=(-1, 1), nrow=im_hq.size(0))
                    writer.add_image(f"{i}", sample, i)

            if (i + 1) % args.save_every == 0:
                torch.save(
                    {
                        "g": g_module.state_dict(),
                        "d": d_module.state_dict(),
                        "g_optim": g_optim.state_dict(),
                        "d_optim": d_optim.state_dict(),
                        "g_sched": g_sched.state_dict(),
                        "d_sched": d_sched.state_dict(),
                        "args": args,
                        "iter": i + 1,
                    },
                    os.path.join(args.output, "latest.pt"),
                )

        if i % 10000 == 0:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()

