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
from models.networks import WIRE, Positional_Encoder, FFN, SIREN
from models.mfn import GaborNet, FourierNet, KGaborNet
from models.wire2d  import WIRE2D
from models.utils import get_config, prepare_sub_folder, get_multiple_slices_dataloader, psnr, ssim, get_device, save_im, stats_per_coil
from metrics.losses import HDRLoss_FF, TLoss, CenterLoss, FocalFrequencyLoss, TanhL2Loss, MSLELoss
from models.regularization import Regularization_L1, Regularization_L2
from log_handler.logger import INRLogger

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='src/config/config_image.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_epoch = config['max_epoch']
in_image_space = config["transform"]
device = get_device(config["model"])
print(device)
cudnn.benchmark = True

# Setup output folder
output_folder = os.path.splitext(os.path.basename(opts.config))[0]
model_name = os.path.join(output_folder, config['data'] + '/img_multi_slices_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}' \
    .format(config['model'], \
        config['net']['network_input_size'], config['net']['network_width'], \
        config['net']['network_depth'], config['loss'], config['lr'], config['encoder']['embedding']))
if not(config['encoder']['embedding'] == 'none'):
    model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
print(model_name)

model_name = model_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
train_writer = INRLogger(os.path.join(opts.output_path + "/logs", model_name))
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder

# Setup loss functions
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
if config['loss'] == 'MSLE':
    loss_fn = MSLELoss()
if config['loss'] == 'T':
    loss_fn = TLoss()
if config['loss'] == 'LSL':
    loss_fn = CenterLoss(config["loss_opts"])
if config['loss'] == 'FFL':
    loss_fn = FocalFrequencyLoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
elif config['loss'] == 'HDR':
    loss_fn = HDRLoss_FF(config['loss_opts'])
elif config['loss'] == 'tanh':
    loss_fn = TanhL2Loss()
else:
    NotImplementedError

# Setup Regularization
# Check do we have regularization
reguluzation_type = config["regularization"]["type"]
if reguluzation_type == "none":
    # if we do not have regularization
    regularization = False
else:
    # if we have regularization
    regularization_strength = config["regularization"]["strenght"]

    regularization_methods = {
        "L1" : Regularization_L1(reg_strength=regularization_strength),
        "L2" : Regularization_L2(reg_strength=regularization_strength)
    }
    regularization = regularization_methods.get(config["regularization"]["type"], None)


if regularization:
    print(f"Regularization is being used {type(regularization)}")
# End of Regularization setup

# Setup data loader
# The only difference of val loader is that data is not shuffled
datasets, data_loaders, val_loaders, all_slices = get_multiple_slices_dataloader(
    data=config['data'],
    data_root=config['data_root'],
    set=config['set'],
    batch_size=config['batch_size'],
    transform=config['transform'],
    num_workers=0,
    sample=config["sample"],
    all_slices= config["all_slices"] , #True,
    shuffle=True,
    full_norm=config["full_norm"],
    normalization=config["normalization"],
    undersampling= config["undersampling"],
    use_dists="no",
    slices=config["slices"]
)

bs = config["batch_size"]

for dataset, data_loader, val_loader, _slice in zip(datasets, data_loaders, val_loaders, all_slices):
    # Setup input encoder:
    encoder = Positional_Encoder(config['encoder'], device=device)

    # Setup model
    if config['model'] == 'SIREN':
        model = SIREN(config['net'])
    elif config['model'] == 'WIRE':
        model = WIRE(config['net'])
    elif config['model'] == 'WIRE2D':
        model = WIRE2D(config['net'])
    elif config['model'] == 'FFN':
        model = FFN(config['net'])
    elif config['model'] == 'Fourier':
        model = FourierNet(config['net'])
    elif config['model'] == 'Gabor':
        model = GaborNet(config['net'])
    elif config['model'] == 'KGabor':
        model = KGaborNet(config['net'])
    else:
        raise NotImplementedError

    # Setup optimizer
    if config['optimizer'] == 'Adam':
        optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']),
                                 weight_decay=config['weight_decay'])
    else:
        NotImplementedError

    if "pretrain" in config:
        checkpoint = torch.load(config["pretrain"], map_location=torch.device(device=device))
        model.load_state_dict(checkpoint["net"])
        optim.load_state_dict(checkpoint["opt"])
        encoder.B = checkpoint["enc"]

    image_shape = dataset.img_shape
    C, H, W, S = image_shape
    print('Load image: {} at slice {}'.format(dataset.file, _slice))

    train_image = torch.zeros(((C*H*W),S)).to(device)

    # Reconstruct image from val # we need to make sure here
    for it, (coords, gt, _) in enumerate(val_loader):
        train_image[it*bs:(it+1)*bs, :] = gt.to(device)

    train_image = train_image.reshape(C,H,W,S).cpu()
    k_space = torch.clone(train_image)
    if not in_image_space: # If in k-space apply inverse fourier trans
        save_im(train_image, image_directory, "slice{}_train_kspace.png".format(_slice), is_kspace=True)
        train_image = fastmri.ifft2c(train_image)
    train_image = fastmri.complex_abs(train_image)
    train_image = fastmri.rss(train_image, dim=0)
    image = torch.clone(train_image)
    save_im(image, image_directory, "slice{}_train.png".format(_slice))

    # torchvision.utils.save_image(normalize_image(torch.abs(train_image),True), os.path.join(image_directory, "train.png"))
    del train_image

    best_psnr = -999999
    best_psnr_ep = 0
    best_ssim = -1
    best_ssim_ep = 0

    print('Training on slice {} for {} epochs'.format(_slice, max_epoch))
    model.to(device=device)
    model.train()
    scheduler = LambdaLR(optim, lambda x: 0.2**min(x/max_epoch, 1))
    for epoch in range(max_epoch):
        model.train()
        running_loss = 0
        for it, (coords, gt, dist_to_center) in enumerate(data_loader):
            # Copy coordinates for HDR loss
            kcoords = torch.clone(coords)
            coords = coords.to(device=device)  # [bs, 3]
            coords = encoder.embedding(coords) # [bs, 2*embedding size]
            gt = gt.to(device=device)  # [bs, 2], [0, 1]
            train_output = None
            if config["model"] == "KGabor":
                train_output = model(coords, dist_to_center)  # [bs, 2]
            else:
                train_output = model(coords)  # [bs, 2]
            optim.zero_grad()
            train_loss = 0
            if config["loss"] in ["HDR", "LSL", "FFL", "tanh"]:
                train_loss, _ = loss_fn(train_output, gt, kcoords.to(device))
            else:
                train_loss = 0.5 * loss_fn(train_output, gt)

            # Regularization check
            if regularization:
                #add refularization term to loss
                train_loss += regularization(model.parameters())

            train_loss.backward()
            optim.step()

            running_loss += train_loss.item()

            if it % config['log_iter'] == config['log_iter'] - 1:
                train_loss = train_loss.item()
                train_writer.log_train(train_loss, epoch * len(data_loader) + it + 1)
                print("[Epoch: {}/{}, Iteration: {}] Train loss: {:.4g}".format(epoch+1, max_epoch, it, train_loss))
                running_loss = 0
        if (epoch + 1) % config['val_epoch'] == 0:
            model.eval()
            test_running_loss = 0
            im_recon = torch.zeros(((C*H*W),S)).to(device)
            with torch.no_grad():
                for it, (coords, gt, dist_to_center) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    kcoords = torch.clone(coords)
                    coords = coords.to(device=device)  # [bs, 3]
                    coords = encoder.embedding(coords) # [bs, 2*embedding size]
                    gt = gt.to(device=device)  # [bs, 2], [0, 1]
                    test_output = None
                    if config["model"] == "KGabor":
                        test_output = model(coords, dist_to_center)  # [bs, 2]
                    else:
                        test_output = model(coords)  # [bs, 2]
                    test_loss = 0
                    if config["loss"] in ["HDR", "LSL", "FFL", "tanh"]:
                        test_loss, _ = loss_fn(test_output, gt, kcoords.to(device))
                    else:
                        test_loss = 0.5 * loss_fn(test_output, gt)
                    test_running_loss += test_loss.item()
                    im_recon[it*bs:(it+1)*bs, :] = test_output
            im_recon = im_recon.reshape(C,H,W,S).detach().cpu()
            if not in_image_space:
                save_im(im_recon.squeeze(), image_directory, "slice{}_recon_kspace_{}dB.png".format(_slice, epoch + 1), is_kspace=True)
                # Plot relative error
                save_im(((im_recon.squeeze() - k_space)), image_directory, "slice{}_recon_kspace_{}_error.png".format(_slice, epoch + 1), is_kspace=True)
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
            save_im(im_recon.squeeze(), image_directory, "slice{}_recon_{}_{:.4g}.png".format(_slice, epoch + 1, test_psnr))
            train_writer.log_test(running_loss/ len(data_loader), test_psnr, test_ssim, epoch+1)
            # Must transfer to .cpu() tensor firstly for saving images
            print("[Validation Epoch: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g} \n Best psnr: {:.4g} @ epoch {} | Best ssim: {:.4g} @ epoch {}"
                  .format(epoch + 1, max_epoch, test_running_loss / len(data_loader), test_psnr, test_ssim, best_psnr, best_psnr_ep, best_ssim, best_ssim_ep))

        if (epoch + 1) % config['image_save_epoch'] == 0:
            # Save final model
            model_name = os.path.join(checkpoint_directory, 'slice{}_'.format(_slice)+'model_%06d.pt' % (epoch + 1))
            torch.save({'net': model.state_dict(), \
                        'enc': encoder.B, \
                        'opt': optim.state_dict(), \
                        }, model_name)
        scheduler.step()