import os
import argparse
import shutil
import yaml
from torch.optim.lr_scheduler import LambdaLR
import torch
import torch.backends.cudnn as cudnn
import fastmri
from datetime import datetime
from tqdm import tqdm
import torch.utils.tensorboard as tensorboardX
from models.utils import get_config, prepare_sub_folder, get_data_loader, psnr, ssim, get_device, save_im, \
    stats_per_coil
from metrics.losses import HDRLoss_FF, TLoss, CenterLoss, FocalFrequencyLoss, TanhL2Loss
from models.networks import Positional_Encoder, WIRE, FFN, SIREN
from models.wire2d import WIRE2D
from parameter_search.find_best_config import grid_search, random_search


def training_function(model_configs, model_name, max_epoch, image_directory, device, image=None, data_loader=None,
                      val_loader=None, image_shape=None):
    # Setup data loader
    # The only difference of val loader is that data is not shuffled
    # loading data
    in_image_space = model_configs["transform"]
    bs = model_configs["batch_size"]
    dataset, data_loader, val_loader = get_data_loader(
        data=model_configs['data'],
        data_root=model_configs['data_root'],
        set=model_configs['set'],
        batch_size=bs,  # config['batch_size'],
        transform=in_image_space,  # config['transform'],
        num_workers=0,
        sample=model_configs["sample"],
        slice=model_configs["slice"],
        shuffle=True,
        full_norm=model_configs["full_norm"],
        normalization=model_configs["normalization"]
    )

    image_shape = dataset.img_shape
    C, H, W, S = image_shape
    print('Load image: {}'.format(dataset.file))

    train_image = torch.zeros(((C * H * W), S)).to(device)
    # Reconstruct image from val
    for it, (coords, gt) in enumerate(val_loader):
        train_image[it * bs:(it + 1) * bs, :] = gt.to(device)
    train_image = train_image.reshape(C, H, W, S).cpu()
    k_space = torch.clone(train_image)
    if not in_image_space and not os.path.exists(
            os.path.join(image_directory, 'train_kspace.png')):  # If in k-space apply inverse fourier trans
        save_im(train_image, image_directory, "train_kspace.png", is_kspace=True)
        train_image = fastmri.ifft2c(train_image)
    train_image = fastmri.complex_abs(train_image)
    train_image = fastmri.rss(train_image, dim=0)
    image = torch.clone(train_image)

    if not os.path.exists(os.path.join(image_directory, 'train.png')):
        save_im(image, image_directory, "train.png")

    # torchvision.utils.save_image(normalize_image(torch.abs(train_image),True), os.path.join(image_directory, "train.png"))
    del train_image

    # Setup input encoder:
    encoder = Positional_Encoder(model_configs['encoder'], device=device)

    bs = model_configs["batch_size"]
    torch.manual_seed(42)
    # Setup model
    if model_name == 'SIREN':
        model = SIREN(model_configs['net'])
    elif model_name == 'WIRE':
        model = WIRE(model_configs['net'])
    elif model_name == 'WIRE2D':
        model = WIRE2D(model_configs['net'])
    elif model_name == 'FFN':
        model = FFN(model_configs['net'])
    else:
        raise NotImplementedError
    in_image_space = model_configs["transform"]

    optim = torch.optim.Adam(model.parameters(), lr=model_configs['lr'],
                             betas=(model_configs['beta1'], model_configs['beta2']),
                             weight_decay=model_configs['weight_decay'])

    # Setup loss functions
    if model_configs['loss'] == 'L2':
        loss_fn = torch.nn.MSELoss()
    if model_configs['loss'] == 'T':
        loss_fn = TLoss()
    if model_configs['loss'] == 'LSL':
        loss_fn = CenterLoss(model_configs["loss_opts"])
    if model_configs['loss'] == 'FFL':
        loss_fn = FocalFrequencyLoss()
    elif model_configs['loss'] == 'L1':
        loss_fn = torch.nn.L1Loss()
    elif model_configs['loss'] == 'HDR':
        loss_fn = HDRLoss_FF(model_configs['loss_opts'])
    elif model_configs['loss'] == 'tanh':
        loss_fn = TanhL2Loss()
    else:
        NotImplementedError

    C, H, W, S = image_shape
    best_psnr = -999999
    best_psnr_ep = 0
    best_ssim = -1
    best_ssim_ep = 0

    model.to(device=device)
    model.train()
    scheduler = LambdaLR(optim, lambda x: 0.2 ** min(x / max_epoch, 1))
    print('Training for {} epochs'.format(max_epoch))
    for epoch in range(max_epoch):
        model.train()
        running_loss = 0
        for it, (coords, gt) in enumerate(data_loader):
            # Copy coordinates for HDR loss
            kcoords = torch.clone(coords)
            coords = coords.to(device=device)  # [bs, 3]
            coords = encoder.embedding(coords)  # [bs, 2*embedding size]
            gt = gt.to(device=device)  # [bs, 2], [0, 1]
            optim.zero_grad()
            train_output = model(coords)  # [bs, 2]
            train_loss = 0
            if model_configs["loss"] in ["HDR", "LSL", "FFL", "tanh"]:
                train_loss, _ = loss_fn(train_output, gt, kcoords.to(device))
            else:
                train_loss = 0.5 * loss_fn(train_output, gt)

            train_loss.backward()
            optim.step()

            running_loss += train_loss.item()

            if it % model_configs['log_iter'] == model_configs['log_iter'] - 1:
                train_loss = train_loss.item()
                # train_writer.add_scalar('train_loss', train_loss / config['log_iter'])
                print("[Epoch: {}/{}, Iteration: {}] Train loss: {:.4g}".format(epoch + 1, max_epoch, it, train_loss))
                running_loss = 0
        if (epoch + 1) % model_configs['val_epoch'] == 0:
            model.eval()
            test_running_loss = 0
            im_recon = torch.zeros(((C * H * W), S)).to(device)
            with torch.no_grad():
                for it, (coords, gt) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    kcoords = torch.clone(coords)
                    coords = coords.to(device=device)  # [bs, 3]
                    coords = encoder.embedding(coords)  # [bs, 2*embedding size]
                    gt = gt.to(device=device)  # [bs, 2], [0, 1]
                    test_output = model(coords)  # [bs, 2]
                    test_loss = 0
                    if model_configs["loss"] in ["HDR", "LSL", "FFL", "tanh"]:
                        test_loss, _ = loss_fn(test_output, gt, kcoords.to(device))
                    else:
                        test_loss = 0.5 * loss_fn(test_output, gt)
                    test_running_loss += test_loss.item()
                    im_recon[it * bs:(it + 1) * bs, :] = test_output
            im_recon = im_recon.reshape(C, H, W, S).detach().cpu()
            if not in_image_space:
                save_im(im_recon.squeeze(), image_directory,
                        "config_{}_recon_kspace_{}dB.png".format(model_configs["config_index"], epoch + 1),
                        is_kspace=True)
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
            # torchvision.utils.save_image(normalize_image(im_recon.squeeze(), True), os.path.join(image_directory, "recon_{}_{:.4g}dB.png".format(epoch + 1, test_psnr)))
            save_im(im_recon.squeeze(), image_directory,
                    "config_{}_recon_{}_{:.4g}.png".format(model_configs["config_index"],
                                                           epoch + 1, test_psnr))
            # train_writer.add_scalar('test_loss', test_running_loss / len(data_loader))
            # train_writer.add_scalar('test_psnr', test_psnr)
            # train_writer.add_scalar('test_ssim', test_ssim)

            # Must transfer to .cpu() tensor firstly for saving images
            print(
                "[Validation Epoch: {}/{}] Test loss: {:.4g} | Test psnr: {:.4g} | Test ssim: {:.4g} \n Best psnr: {:.4g} @ epoch {} | Best ssim: {:.4g} @ epoch {}"
                .format(epoch + 1, max_epoch, test_running_loss / len(data_loader), test_psnr, test_ssim, best_psnr,
                        best_psnr_ep, best_ssim, best_ssim_ep))

        # if (epoch + 1) % config['image_save_epoch'] == 0:
        # Save final model
        # model_name = os.path.join(checkpoint_directory, 'model_%06d.pt' % (epoch + 1))
        # torch.save({'net': model.state_dict(), 'enc': encoder.B, 'opt': optim.state_dict(), }, model_name)
        scheduler.step()

    return model, {"best_psnr": best_psnr, "best_psnr_ep": best_psnr_ep, "best_ssim": best_ssim,
                   "best_ssim_ep": best_ssim_ep}


def run(config, hp_config):
    max_epoch = config.pop('max_epoch')
    device = get_device(config["model"])
    print("Running on: ", device)
    cudnn.benchmark = True

    # Setup output folder
    output_folder = os.path.splitext(os.path.basename(opts.config))[0]
    model_name = os.path.join(output_folder, config['data'] + '/img_{}_{}_{}_{}_{}_lr{:.2g}_encoder_{}_hp_{}_search_' \
                              .format(config['model'], config['net']['network_input_size'],
                                      config['net']['network_width'], config['net']['network_depth'], config['loss'],
                                      config['lr'], config['encoder']['embedding'],
                                      hp_config["method"]))
    if not (config['encoder']['embedding'] == 'none'):
        model_name += '_scale{}_size{}'.format(config['encoder']['scale'], config['encoder']['embedding_size'])
    print(model_name)

    train_writer = tensorboardX.SummaryWriter(os.path.join(opts.output_path + "/logs", model_name))
    output_directory = os.path.join(opts.output_path + "/outputs",
                                    model_name + datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    checkpoint_directory, image_directory = prepare_sub_folder(output_directory)
    shutil.copy(opts.config, os.path.join(output_directory, 'config.yaml'))  # copy config file to output folder

    ## adding useful additional_data to configs, which are popped before running the best configs.
    config["image_directory"] = image_directory
    config["output_directory"] = output_directory

    search_method = hp_config.pop("method")
    if search_method == "grid":
        print("** Running Grid Search **")
        best_configs = grid_search(train_function=training_function, model_configs=config, model_class=config["model"],
                                   epochs=hp_config.pop("max_epoch"), grid_search_spaces=hp_config.pop("search_space"),
                                   device=device)

    else:
        print("** Running Random Search **")
        best_configs = random_search(train_function=training_function, model_configs=config,
                                     model_class=config["model"],
                                     num_search=hp_config.pop("num_search"), epochs=hp_config.pop("max_epoch"),
                                     random_search_spaces=hp_config.pop("search_space"), device=device)

    with open(os.path.join(output_directory, "best_psnr_config.yaml"), "w") as yaml_file:
        yaml.dump(best_configs["PSNR"]["config"], yaml_file, default_flow_style=False)

    with open(os.path.join(output_directory, "best_ssim_config.yaml"), "w") as yaml_file:
        yaml.dump(best_configs["SSIM"]["config"], yaml_file, default_flow_style=False)

    with open(os.path.join(output_directory, "configs_and_results.txt"), "w") as tf:
        for item in best_configs["results"]:
            tf.write("{} -> {}\n".format(item[0], item[1]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='src/config/config_image.yaml', help='Path to the config file.')
    parser.add_argument('--hp_config', type=str, default='src/hp_config/config_image.json',
                        help='Path to the HP config file.')
    parser.add_argument('--output_path', type=str, default='.', help="outputs path")
    # Load experiment setting
    opts = parser.parse_args()
    config = get_config(opts.config)
    hp_config = get_config(opts.hp_config)

    # Running the hyperparameter search algorithm
    run(config=config, hp_config=hp_config)
