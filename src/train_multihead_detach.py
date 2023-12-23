import os
import argparse
import shutil
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.backends.cudnn as cudnn
import fastmri
from datetime import datetime
from tqdm import tqdm
import torch.utils.tensorboard as tensorboardX
from models.networks import WIRE, Positional_Encoder, FFN, SIREN, MultiHeadWrapper
from models.wire2d  import WIRE2D
from models.utils import get_config, prepare_sub_folder, get_data_loader, psnr, ssim, get_device, save_im, stats_per_coil
from metrics.losses import HDRLoss_FF, TLoss, CenterLoss, FocalFrequencyLoss, TanhL2Loss
from math import sqrt
from clustering import partition_and_stats
import numpy as np

def train(opts):
    # Load experiment setting
    config = get_config(opts.config)
    max_epoch = config['max_epoch']
    in_image_space = config["transform"]
    device = get_device(config["model"])
    print(device)
    cudnn.benchmark = True

    # Setup output folder
    output_folder = os.path.splitext(os.path.basename(opts.config))[0]
    model_name = os.path.join(output_folder, config['data'] + '/img_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
        .format(config['model'], \
            config['net']['network_input_size'], config['net']['network_width'], \
            config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding']))
    if not(config['encoder']['embedding'] == 'none'):
        model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
    print(model_name)

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name  + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


    # Setup input encoder:
    encoder = Positional_Encoder(config['encoder'], device=device)

    # Setup model
    if config['model'] == 'SIREN':
        model_back = SIREN(config['net'])
    elif config['model'] == 'WIRE':
        model_back = WIRE(config['net'])
    elif config['model'] == 'WIRE2D':
        model_back = WIRE2D(config['net'])
    elif config['model'] == 'FFN':
        model_back = FFN(config['net'])
    else:
        raise NotImplementedError
    model_back.to(device=device)
    model_back.train()
    model_back = None

    part_config = config["partition"]
    no_models = part_config["no_models"]
    no_steps = part_config["no_steps"]
    expand = part_config["expansion"]

    model = MultiHeadWrapper(
        backbone=model_back,
        no_heads=no_models,
        params=config["subnets"],
        device=device,
        detach_outs=True
    )

    params = []
    if model.backbone:
        params = params + list(model.backbone.parameters())
    for i in range(no_models):
        params = params + list(model.heads[i].parameters())
    params = params + list(model.weighted_avg.parameters())

    # Setup optimizer
    if config['optimizer'] == 'Adam':
        # Make sure all params optimized
        optim = torch.optim.Adam(params, lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
    else:
        NotImplementedError

    # Setup loss functions
    if config['loss'] == 'L2':
        loss_fn = torch.nn.MSELoss()
    if config['loss'] == 'smoothL1':
        loss_fn = torch.nn.SmoothL1Loss()
    if config['loss'] == 'T':
        loss_fn = TLoss()
    if config['loss'] == 'LSL':
        loss_fn = CenterLoss(config["loss_opts"])
    if config['loss'] == 'FFL':
        loss_fn = FocalFrequencyLoss(config=config["loss_opts"])
    elif config['loss'] == 'L1':
        loss_fn = torch.nn.L1Loss()
    elif config['loss'] == 'HDR':
        loss_fn = HDRLoss_FF(config['loss_opts'])
    elif config['loss'] == 'tanh':
        loss_fn = TanhL2Loss()
    else:
        NotImplementedError

    if "pretrain" in config:
        checkpoint = torch.load(config["pretrain"], map_location=torch.device(device=device))
        model.load_state_dict(checkpoint["net"])
        optim.load_state_dict(checkpoint["opt"])
        encoder.B = checkpoint["enc"]

    # Setup data loader
    # The only difference of val loader is that data is not shuffled
    dataset, data_loader, val_loader = get_data_loader(
        data=config['data'], 
        data_root=config['data_root'], 
        set=config['set'], 
        batch_size=config['batch_size'],
        transform=config['transform'], 
        num_workers=2, 
        sample=config["sample"], 
        slice=config["slice"],
        shuffle=True,
        full_norm=config["full_norm"],
        normalization=config["normalization"],
        use_dists="yes"
    )

    mx, part_radii = partition_and_stats(
        dataset=dataset, 
        no_steps=part_config["no_steps"],
        no_parts=part_config["no_models"],
        show=False,
    )
    mx = torch.pow(mx,1/3)
    print("Kmeans Radial partitioning:")
    print(part_radii / sqrt(2))
    print("Scaled Max per Cluster:")
    print(mx)


    bs = config["batch_size"]
    image_shape = dataset.img_shape
    C, H, W, S = image_shape
    print('Load image: {}'.format(dataset.file))

    train_image = torch.zeros(((C*H*W),S)).to(device)
    # Reconstruct image from val
    for it, (_, gt, _) in enumerate(val_loader):
        train_image[it*bs:(it+1)*bs, :] = gt.to(device)
    train_image = train_image.reshape(C,H,W,S).cpu()
    k_space = torch.clone(train_image)
    if not in_image_space: # If in k-space apply inverse fourier trans
        save_im(train_image, image_directory, "train_kspace.png", is_kspace=True)
        train_image = fastmri.ifft2c(train_image)
    train_image = fastmri.complex_abs(train_image)
    train_image = fastmri.rss(train_image, dim=0)
    image = torch.clone(train_image)
    save_im(image, image_directory, "train.png")

    # torchvision.utils.save_image(normalize_image(torch.abs(train_image),True), os.path.join(image_directory, "train.png"))
    del train_image

    best_psnr = -999999
    best_psnr_ep = 0
    best_ssim = -1
    best_ssim_ep = 0

    scheduler = LambdaLR(optim, lambda x: 0.2**min(x/max_epoch, 1))
    print('Training for {} epochs'.format(max_epoch))
    for epoch in range(max_epoch):
        model.train()
        running_loss = 0
        for it, (coords, gt, dist_to_center) in enumerate(data_loader):
            # Copy coordinates for HDR loss
            coords, gt = coords.to(device), gt.to(device)
            dist_to_center = dist_to_center.to(device)
            coords = encoder.embedding(coords) # [bs, 2*embedding size]
            layer_outs, train_output = model(coords, dist_to_center)
            train_loss = 0
            for i in range(no_models):
                r_0 = part_radii[i] - expand
                r_1 = part_radii[i+1] + expand
                ind = torch.where((dist_to_center >= r_0) & (dist_to_center <= r_1))
                if ind[0].numel():
                    gt_local = gt[ind]
                    for idx, out in enumerate(layer_outs):
                        # Get gradients for final layers, and scale if target
                        # Make hyperparam is better probs
                        out_local = out[ind]
                        # Renormalize with 1!
                        multiplier = (1 if idx == i else 0.00000001)/mx[i]
                        if config["loss"] in ["HDR", "LSL", "FFL", "tanh"]:
                            loss, _ = loss_fn(out_local, gt_local, coords.to(device))
                            train_loss += multiplier * loss
                        else:
                            train_loss += multiplier * loss_fn(out_local, gt_local)
            if config["loss"] in ["HDR", "LSL", "FFL", "tanh"]:
                loss, _ = loss_fn(train_output, gt, coords.to(device))
                train_loss += loss
            else:
                train_loss +=  loss_fn(train_output, gt)
            optim.zero_grad()
            train_loss.backward()
            optim.step()
            running_loss += train_loss.item()
            # Enforce that each weight should have at least a minimal positive contribution

            if it % config['log_iter'] == config['log_iter'] - 1:
                train_loss = train_loss.item()
                train_writer.add_scalar('train_loss', train_loss/config['log_iter'])
                print("[Epoch: {}/{}, Iteration: {}] Train loss: {:.4g}".format(epoch+1, max_epoch, it, train_loss))
                running_loss = 0
        if (epoch + 1) % config['val_epoch'] == 0:
            model.eval()
            test_running_loss = 0
            im_recon = torch.zeros(((C*H*W),S))
            with torch.no_grad():
                # Separation of indices not needed for val
                for it, (coords, gt, dist_to_center) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    coords, gt = coords.to(device), gt.to(device)
                    coords = encoder.embedding(coords) # [bs, 2*embedding size]
                    dist_to_center = dist_to_center.to(device)
                    _,test_output = model(coords, dist_to_center)
                    test_loss = 0
                    if config["loss"] in ["HDR", "LSL", "FFL", "tanh"]:
                        test_loss, _ = loss_fn(test_output, gt, coords.to(device))
                    else:
                        test_loss = loss_fn(test_output, gt)
                    test_running_loss += test_loss.item()
                    im_recon[it*bs:(it+1)*bs, :] = test_output.detach().cpu()
            im_recon = im_recon.view(C,H,W,S)

            if not in_image_space:
                save_im(im_recon.squeeze(), image_directory, "recon_kspace_{}dB.png".format(epoch + 1), is_kspace=True)
                # Plot relative error
                save_im(((im_recon.squeeze() - k_space)), image_directory, "recon_kspace_{}_error.png".format(epoch + 1), is_kspace=True)
                stats_per_coil(im_recon, C)
                im_recon = fastmri.ifft2c(im_recon)
            im_recon = fastmri.complex_abs(im_recon)
            im_recon = fastmri.rss(im_recon, dim=0)
            test_psnr = psnr(image, im_recon).item() 
            test_ssim = ssim(image, im_recon).item() 
            if test_psnr > best_psnr:
                best_psnr = test_psnr
                best_psnr_ep = epoch
            if test_ssim > best_ssim:
                best_ssim = test_ssim
                best_ssim_ep = epoch
            # torchvision.utils.save_image(normalize_image(im_recon.squeeze(), True), os.path.join(image_directory, "recon_{}_{:.4g}dB.png".format(epoch + 1, test_psnr)))
            save_im(im_recon.squeeze(), image_directory, "recon_{}_{:.4g}.png".format(epoch + 1, test_psnr))
            train_writer.add_scalar('test_loss', test_running_loss / len(data_loader))
            train_writer.add_scalar('test_psnr', test_psnr)
            train_writer.add_scalar('test_ssim', test_ssim)
            # Must transfer to .cpu() tensor firstly for saving images
            print("[Validation Epoch: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g} \n Best psnr: {:.4g} @ epoch {} | Best ssim: {:.4g} @ epoch {}"
                .format(epoch + 1, max_epoch, test_running_loss / len(data_loader), test_psnr, test_ssim, best_psnr, best_psnr_ep, best_ssim, best_ssim_ep))
            del im_recon

        if (epoch + 1) % config['image_save_epoch'] == 0:
            # Save final model
            model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (epoch + 1))
            torch.save({'net': model.state_dict(), \
                        'enc': encoder.B, \
                        'opt': optim.state_dict(), \
                        }, model_name)
        scheduler.step()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/config_image.yaml', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")

    # Load experiment setting
    opts = parser.parse_args()
    train(opts)