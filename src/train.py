import os
import argparse
import shutil
from torch.optim.lr_scheduler import LambdaLR
import torch
import torchvision
import torch.backends.cudnn as cudnn
import fastmri
import torch.utils.tensorboard as tensorboardX
import matplotlib.pyplot as plt
from models.networks import WIRE, Positional_Encoder, FFN, SIREN
from models.wire2d  import WIRE2D
import numpy as np
from models.utils import get_config, prepare_sub_folder, get_data_loader, save_image_3d, psnr, ssim, get_device

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='src/config/config_image.yaml', help='Path to the config file.')
parser.add_argument('--output_path', type=str, default='.', help="outputs path")

# Load experiment setting
opts = parser.parse_args()
config = get_config(opts.config)
max_epoch = config['max_epoch']
in_image_space = config["transform"]
device = get_device(config["model"])
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
output_directory = os.path.join(opts.output_path + "/outputs", model_name)
checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml')) # copy config file to output folder


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
else:
    raise NotImplementedError
model.to(device=device)
model.train()

# Setup optimizer
if config['optimizer'] == 'Adam':
    optim = torch.optim.Adam(model.parameters(), lr=config['lr'], betas=(config['beta1'], config['beta2']), weight_decay=config['weight_decay'])
else:
    NotImplementedError

# Setup loss functions
if config['loss'] == 'L2':
    loss_fn = torch.nn.MSELoss()
elif config['loss'] == 'L1':
    loss_fn = torch.nn.L1Loss()
else:
    NotImplementedError


# Setup data loader
# The only difference of val loader is that data is not shuffled
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
    full_norm=config["full_norm"]
)

bs = config["batch_size"]
image_shape = dataset.img_shape
C, H, W, S = image_shape
print('Load image: {}'.format(dataset.file))

train_image = torch.zeros(((C*H*W),S)).to(device)
# Reconstruct image from val
for it, (coords, gt) in enumerate(val_loader):
    train_image[it*bs:(it+1)*bs, :] = gt.to(device)
train_image = train_image.reshape(C,H,W,S).cpu()
if not in_image_space: # If in k-space apply inverse fourier trans
    train_image = fastmri.ifft2c(train_image)
train_image = fastmri.complex_abs(train_image)
train_image = fastmri.rss(train_image, dim=0)
image = torch.clone(train_image)
plt.imshow(np.abs(image.numpy()), cmap='gray')
plt.savefig(os.path.join(image_directory, "train.png"))
plt.clf()

# torchvision.utils.save_image(normalize_image(torch.abs(train_image),True), os.path.join(image_directory, "train.png"))
del train_image

scheduler = LambdaLR(optim, lambda x: 0.2**min(x/max_epoch, 1))
print('Training for {} epochs'.format(max_epoch))
for epoch in range(max_epoch):
    model.train()
    running_loss = 0
    for it, (coords, gt) in enumerate(data_loader):
        coords = coords.to(device=device)  # [bs, 3]
        coords = encoder.embedding(coords) # [bs, 2*embedding size]
        gt = gt.to(device=device)  # [bs, 2], [0, 1]
        optim.zero_grad()
        train_output = model(coords)  # [bs, 2]
        train_loss = 0.5 * loss_fn(train_output, gt)

        train_loss.backward()
        optim.step()

        running_loss += train_loss.item()

        if it % config['log_iter'] == config['log_iter'] - 1:
            train_loss = train_loss.item()
            train_writer.add_scalar('train_loss', train_loss/config['log_iter'])
            print("[Epoch: {}/{}, Iteration: {}] Train loss: {:.4g}".format(epoch+1, max_epoch, it, train_loss))
            running_loss = 0
    if (epoch + 1) % config['val_epoch'] == 0:
        model.eval()
        test_running_loss = 0
        im_recon = torch.zeros(((C*H*W),S)).to(device)
        with torch.no_grad():
            for it, (coords, gt) in enumerate(val_loader):
                coords = coords.to(device=device)  # [bs, 3]
                coords = encoder.embedding(coords) # [bs, 2*embedding size]
                gt = gt.to(device=device)  # [bs, 2], [0, 1]
                test_output = model(coords)  # [bs, 2]
                test_loss = 0.5 * loss_fn(test_output, gt)
                test_running_loss += test_loss.item()
                im_recon[it*bs:(it+1)*bs, :] = test_output
        im_recon = im_recon.reshape(C,H,W,S).detach().cpu()
        if not in_image_space:
            im_recon = fastmri.ifft2c(im_recon)
        im_recon = fastmri.complex_abs(im_recon)
        im_recon = fastmri.rss(im_recon, dim=0)
        test_psnr = psnr(image, im_recon).item() 
        test_ssim = ssim(image, im_recon).item() 
        # torchvision.utils.save_image(normalize_image(im_recon.squeeze(), True), os.path.join(image_directory, "recon_{}_{:.4g}dB.png".format(epoch + 1, test_psnr)))
        plt.imshow(np.abs(im_recon.squeeze().numpy()), cmap='gray')
        plt.savefig(os.path.join(image_directory, "recon_{}_{:.4g}dB.png".format(epoch + 1, test_psnr)))
        plt.clf()
        train_writer.add_scalar('test_loss', test_running_loss / len(data_loader))
        train_writer.add_scalar('test_psnr', test_psnr)
        train_writer.add_scalar('test_ssim', test_ssim)
        # Must transfer to .cpu() tensor firstly for saving images
        print("[Validation Epoch: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g}".format(epoch + 1, max_epoch, test_loss, test_psnr, test_ssim))

    if (epoch + 1) % config['image_save_epoch'] == 0:
        # Save final model
        model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (epoch + 1))
        torch.save({'net': model.state_dict(), \
                    'enc': encoder.B, \
                    'opt': optim.state_dict(), \
                    }, model_name)
    scheduler.step()
