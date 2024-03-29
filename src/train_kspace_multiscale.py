"""

Multi-scale KSpace training based on distance from origin

This Training script uses the MultiscaleFourier class,
which is based on BACON and uses a similar notion. In a similar way,
we aim at limiting the information learned at each stage of the network.

We use k-means clustering to produce rings to focus at each 
stage. Since k-space data tends to cluster high intensity points in the center
and get weaker the further you go away from the center we get
the desired limiting of the intensity.

"""
import os
import argparse
import shutil
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.backends.cudnn as cudnn
import fastmri
from datetime import datetime
from tqdm import tqdm
from models.networks import Positional_Encoder
from models.mfn import MultiscaleBoundedFourier, MultiscaleKFourier
from models.utils import get_config, prepare_sub_folder, get_data_loader, psnr, ssim, get_device, save_im, \
    stats_per_coil, get_multiple_slices_dataloader
from metrics.losses import ConsistencyLoss, HDRLoss_FF, LogSpaceLoss, tv_loss
from clustering import partition_and_stats
from log_handler.logger import INRLogger
from utils import set_default_configs


def limit_kspace(kspace, dist, bounds):
    ind = torch.where((dist < bounds[0])
                      | (dist > bounds[1]))
    limited = torch.clone(kspace)
    limited[ind == 0]
    return limited


def create_pairs(values, multiplication_factor):
    pairs = [(values[0], values[i + 1]) for i in range(len(values) - 1)]
    # Cover whole k-space with last one
    # pairs.append((0,5))
    repeated_pairs = [(pair[0], pair[1]) for pair in pairs for _ in range(multiplication_factor)]
    return repeated_pairs


def training_multiscale(config, dataset, data_loader, val_loader, sample, slice_no):
    max_epoch = config['max_epoch']
    in_image_space = config["transform"]
    device = get_device(config["model"])
    print(device)
    cudnn.benchmark = True

    # Setup output folder
    output_folder = os.path.splitext(os.path.basename(opts.config))[0]
    model_name = os.path.join(output_folder, config['data'] + '/img_sample{}_slice{}_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
                              .format(sample, slice_no, config['model'], \
                                      config['net']['network_input_size'], config['net']['network_width'], \
                                      config['net']['network_depth'], config['loss'], config['lr'],
                                      config['encoder']['embedding']))
    if not (config['encoder']['embedding'] == 'none'):
        model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
    print(model_name)
    model_name = model_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    train_writer = INRLogger(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs", model_name)
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    part_config = config["partition"]
    mx, part_radii = partition_and_stats(
        dataset=dataset,
        no_steps=part_config["no_steps"],
        no_parts=part_config["no_models"],
        stat="max",
        show=False,
    )
    mx = torch.cat((mx, torch.Tensor([1])))
    print("stats")
    print(mx)
    pairs = create_pairs(part_radii, 1)
    pairs_model = create_pairs(part_radii, 2)
    print("Used pairs:")
    print(pairs)

    # Setup input encoder:
    encoder = Positional_Encoder(config['encoder'], device=device)

    # Setup model
    model = None
    if config['model'] == 'BoundedFourier':
        print("Bounded")
        model = MultiscaleBoundedFourier(config["net"], boundaries=pairs_model)
    else:
        print("Unbounded")
        model = MultiscaleKFourier(config["net"])
    model.to(device=device)
    model.train()

    # Setup optimizer
    if config['optimizer'] == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                                 weight_decay=config['weight_decay'])
    else:
        NotImplementedError

    # Setup loss functions
    if config['loss'] == 'L2':
        loss_fn = torch.nn.MSELoss()
    if config['loss'] == 'LSL':
        loss_fn = LogSpaceLoss(config["loss_opts"])
    elif config['loss'] == 'L1':
        loss_fn = torch.nn.L1Loss()
    elif config['loss'] == 'HDR':
        loss_fn = HDRLoss_FF(config['loss_opts'])
    else:
        NotImplementedError

    loss_cons = ConsistencyLoss(pairs)
    if "pretrain" in config:
        checkpoint = torch.load(config["pretrain"], map_location=torch.device(device=device))
        model.load_state_dict(checkpoint["net"])
        optim.load_state_dict(checkpoint["opt"])
        encoder.B = checkpoint["enc"]

    # Small first
    # lims = torch.flip(lims,dims=(0,))

    bs = config["batch_size"]
    image_shape = dataset.img_shape
    C, H, W, S = image_shape
    print('Load image: {}'.format(dataset.file))

    train_image = torch.zeros(((C * H * W), S)).to(device)
    # Reconstruct image from val
    for it, (coords, gt, _, _) in enumerate(val_loader):
        train_image[it * bs:(it + 1) * bs, :] = gt.to(device)
    train_image = train_image.reshape(C, H, W, S).cpu()
    k_space = torch.clone(train_image)
    if not in_image_space:  # If in k-space apply inverse fourier trans
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

    scheduler = LambdaLR(optim, lambda x: 0.2 ** min(x / max_epoch, 1))
    print('Training for {} epochs'.format(max_epoch))
    for epoch in range(max_epoch):
        model.train()
        running_loss = 0
        for it, (coords, gt, dist_to_center, mask_coords) in enumerate(data_loader):
            # Copy coordinates for HDR loss
            coords = coords.to(device=device)  # [bs, 3]
            dist_to_center = dist_to_center.to(device)
            gt = gt.to(device=device)  # [bs, 2], [0, 1]
            coords = encoder.embedding(coords) # [bs, 2*embedding size]
            train_output = model(coords=coords, dist_to_center=dist_to_center)  # [bs, 2]
            optim.zero_grad()
            train_loss = 0
            if config["use_tv"]:
                mask_coords = mask_coords.to(device=device)
                train_loss += tv_loss(train_output[-1].view((H,W,2)))
            if len(mask_coords) != 0:
                mask_coords = mask_coords.to(device=device)
                gt = gt[mask_coords[:,0]]
            train_loss += 0.1 * loss_cons(train_output, dist_to_center)
            for idx, out in enumerate(train_output):
                if len(mask_coords) != 0:
                    out = out[mask_coords[:,0]]
                if config["loss"] in ["HDR", "FFL", "tanh"]:
                    # loss, _ = loss_fn(out, limit_kspace(gt, dist_to_center, pairs[idx]), gt) / mx[idx]
                    loss, _ = loss_fn(out, limit_kspace(gt, dist_to_center, pairs[idx]), gt)

                    train_loss += loss / mx[idx]
                else:
                    # train_loss = 0.5 * loss_fn(out, limit_kspace(gt, dist_to_center, pairs[idx])) / mx[idx]
                    train_loss += 0.5 * loss_fn(out, limit_kspace(gt, dist_to_center, pairs[idx]))

            train_loss.backward()
            optim.step()

            running_loss += train_loss.item()

            if it % config['log_iter'] == config['log_iter'] - 1:
                train_loss = train_loss.item()
                train_writer.log_train(train_loss, epoch * len(data_loader) + it + 1)
                print("[Epoch: {}/{}, Iteration: {}] Train loss: {:.4g}".format(epoch + 1, max_epoch, it, train_loss))
                running_loss = 0
        if (epoch + 1) % config['val_epoch'] == 0:
            model.eval()
            test_running_loss = 0
            im_recon = torch.zeros(((C * H * W), S)).to(device)
            with torch.no_grad():
                for it, (coords, gt, dist_to_center, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    coords = coords.to(device=device)  # [bs, 3]
                    dist_to_center = dist_to_center.to(device)
                    coords = encoder.embedding(coords)  # [bs, 2*embedding size]
                    gt = gt.to(device=device)  # [bs, 2], [0, 1]
                    test_output = model(coords=coords, dist_to_center=dist_to_center)  # [bs, 2]

                    test_loss = 0
                    # test_loss += torch.nn.functional.mse_loss(test_output[-1],gt)
                    for idx, out in enumerate(test_output):
                        if config["loss"] in ["HDR", "FFL", "tanh"]:
                            # test_loss, _ = loss_fn(out, limit_kspace(gt, dist_to_center, pairs[idx]), gt) / mx[idx]
                            test_loss, _ = loss_fn(out, limit_kspace(gt, dist_to_center, pairs[idx]), gt)

                        else:
                            # test_loss = 0.5 * loss_fn(out, limit_kspace(gt, dist_to_center, pairs[idx])) / mx[idx]
                            test_loss = 0.5 * loss_fn(out, limit_kspace(gt, dist_to_center, pairs[idx]))
                    test_running_loss += test_loss.item()
                    im_recon[it * bs:(it + 1) * bs, :] = test_output[-1]
            im_recon = im_recon.reshape(C, H, W, S).detach().cpu()
            if not in_image_space:
                save_im(im_recon.squeeze(), image_directory, "recon_kspace_{}dB.png".format(epoch + 1), is_kspace=True)
                # Plot relative error
                save_im(((im_recon.squeeze() - k_space)), image_directory,
                        "recon_kspace_{}_error.png".format(epoch + 1), is_kspace=True)
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
            save_im(im_recon.squeeze(), image_directory, "recon_{}_{:.4g}_psnr_{:.4g}_ssim.png".format(epoch + 1, test_psnr, test_ssim))
            train_writer.log_test(running_loss / len(data_loader), test_psnr, test_ssim, epoch + 1)
            # Must transfer to .cpu() tensor firstly for saving images
            print(
                "[Validation Epoch: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g} \n Best psnr: {:.4g} @ epoch {} | Best ssim: {:.4g} @ epoch {}"
                .format(epoch + 1, max_epoch, test_running_loss / len(data_loader), test_psnr, test_ssim, best_psnr,
                        best_psnr_ep, best_ssim, best_ssim_ep))

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
    parser.add_argument('--data_samples', type=str, default='', help='Path to the config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")

    # Load experiment setting
    opts = parser.parse_args()
    config = get_config(opts.config)

    # Set to normal per-point mode if coil batching not present
    config = set_default_configs(config)
    
    data_samples = get_config(opts.data_samples)

    # Setup data loader
    # The only difference of val loader is that data is not shuffled

    if not data_samples:
        dataset, data_loader, val_loader = get_data_loader(
            data=config['data'],
            data_root=config['data_root'],
            set=config['set'],
            batch_size=config['batch_size'],
            transform=config['transform'],
            num_workers=0,
            sample=config["sample"],
            slice=config["slice"],
            shuffle=True,
            full_norm=config["full_norm"],
            normalization=config["normalization"],
            undersampling=config["undersampling"],
            use_dists="yes",
            per_coil=config["per_coil"]
        )

        training_multiscale(config=config, dataset=dataset, data_loader=data_loader,
                            val_loader=val_loader, sample=config["sample"], slice_no=config["slice"])

    else:
        samples = data_samples["samples"]
        print("Samples: ", samples)
        for sample, slices in samples.items():


            datasets, data_loaders, val_loaders = get_multiple_slices_dataloader(
                data=config['data'],
                data_root=config['data_root'],
                set=config['set'],
                batch_size=config['batch_size'],
                transform=config['transform'],
                num_workers=0,
                sample=config["sample"],
                all_slices=False,
                slices=slices,
                shuffle=True,
                full_norm=config["full_norm"],
                normalization=config["normalization"],
                undersampling=config["undersampling"],
                use_dists="yes",
                per_coil=config["per_coil"]
            )

            for dataset, data_loader, val_loader, _slice in zip(datasets, data_loaders, val_loaders, slices):
                training_multiscale(config=config, dataset=dataset, data_loader=data_loader,
                                    val_loader=val_loader, sample=sample, slice_no=_slice)

            del datasets, data_loaders, val_loaders
