import os
from torch.optim.lr_scheduler import LambdaLR
import torch
import fastmri
from tqdm import tqdm
from src.models.utils import psnr, ssim, save_im, stats_per_coil
from src.metrics.losses import HDRLoss_FF, TLoss, CenterLoss, FocalFrequencyLoss, TanhL2Loss, MSLELoss, tv_loss
from src.models.networks import Positional_Encoder, WIRE, FFN, SIREN
from src.models.mfn import GaborNet, FourierNet, KGaborNet
from src.models.wire2d import WIRE2D


def hp_training_function(config, max_epoch, image_directory, device, # image=None,
                         dataset=None, data_loader=None, val_loader=None):
    in_image_space = config["transform"]
    bs = config["batch_size"]

    image_shape = dataset.img_shape
    C, H, W, S = image_shape
    print('Load image: {}'.format(dataset.file))

    train_image = torch.zeros(((C * H * W), S)).to(device)
    # Reconstruct image from val
    for it, (coords, gt, _, _) in enumerate(val_loader):
        # import pdb; pdb.set_trace()
        train_image[it * bs:(it + 1) * bs, :] = gt.to(device)

    train_image = train_image.reshape(C, H, W, S).cpu()
    k_space = torch.clone(train_image)
    if not in_image_space:  # If in k-space apply inverse fourier trans
        save_im(train_image, image_directory, "train_kspace.png", is_kspace=True)
        train_image = fastmri.ifft2c(train_image)
    train_image = fastmri.complex_abs(train_image)
    train_image = fastmri.rss(train_image, dim=0)
    image = torch.clone(train_image)

    if not os.path.exists(os.path.join(image_directory, 'train.png')):
        save_im(image, image_directory, "train.png")

    del train_image
    # save_im(image, image_directory, "train.png")

    # torchvision.utils.save_image(normalize_image(torch.abs(train_image),True), os.path.join(image_directory, "train.png"))

    # Setup input encoder:
    encoder = Positional_Encoder(config['encoder'], device=device)

    bs = config["batch_size"]
    torch.manual_seed(42)
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

    in_image_space = config["transform"]

    optim = torch.optim.Adam(model.parameters(), lr=config['lr'],
                             betas=(config['beta1'], config['beta2']),
                             weight_decay=config['weight_decay'])

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
    regularization_type = config["regularization"]["type"]
    if regularization_type == "none":
        regularization = False
    else:
        regularization_strength = config["regularization"]["strenght"]

        from src.models.regularization import Regularization_L1, Regularization_L2
        regularization_methods = {
            "L1": Regularization_L1(reg_strength=regularization_strength),
            "L2": Regularization_L2(reg_strength=regularization_strength)
        }
        regularization = regularization_methods.get(config["regularization"]["type"], None)

    if regularization:
        print(f"Regularization is being used {type(regularization)}")

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
        for it, (coords, gt, dist_to_center, mask_coords) in enumerate(data_loader):
            # Copy coordinates for HDR loss
            kcoords = torch.clone(coords)
            coords = coords.to(device=device)  # [bs, 3]
            gt = gt.to(device=device)  # [bs, 2], [0, 1]
            coords = encoder.embedding(coords)  # [bs, 2*embedding size]

            train_output = None
            if config["model"] == "KGabor":
                train_output = model(coords, dist_to_center)  # [bs, 2]
            else:
                train_output = model(coords)  # [bs, 2]
            optim.zero_grad()
            train_loss = 0
            if len(mask_coords) != 0:
                if config["use_tv"]:
                    train_loss += tv_loss(train_output.view((H, W, 2)))
                mask_coords.to(device=device)
                train_output = train_output[mask_coords[:, 0]]
                gt = gt[mask_coords[:, 0]]
            if config["loss"] in ["HDR", "LSL", "FFL", "tanh"]:
                loss, _ = loss_fn(train_output, gt, kcoords.to(device))
                train_loss += loss
            else:
                train_loss += 0.5 * loss_fn(train_output, gt)

            # Regularization check
            if regularization:
                # add refularization term to loss
                train_loss += regularization(model.parameters())

            train_loss.backward()
            optim.step()

            running_loss += train_loss.item()

            if it % config['log_iter'] == config['log_iter'] - 1:
                train_loss = train_loss.item()
                # train_writer.log_train(train_loss, epoch * len(data_loader) + it + 1)
                print("[Epoch: {}/{}, Iteration: {}] Train loss: {:.4g}".format(epoch + 1, max_epoch, it, train_loss))
                running_loss = 0
        if (epoch + 1) % config['val_epoch'] == 0:
            model.eval()
            test_running_loss = 0
            im_recon = torch.zeros(((C * H * W), S)).to(device)
            with torch.no_grad():
                for it, (coords, gt, dist_to_center, _) in tqdm(enumerate(val_loader), total=len(val_loader)):
                    kcoords = torch.clone(coords)
                    coords = coords.to(device=device)  # [bs, 3]
                    coords = encoder.embedding(coords)  # [bs, 2*embedding size]
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
                    im_recon[it * bs:(it + 1) * bs, :] = test_output
            im_recon = im_recon.reshape(C, H, W, S).detach().cpu()
            if not in_image_space:
                save_im(im_recon.squeeze(), image_directory,
                        "config_{}_recon_kspace_{}dB.png".format(config["config_index"], epoch + 1),
                        is_kspace=True)
                # Plot relative error
                save_im(((im_recon.squeeze() - k_space)), image_directory,
                        "config_{}_recon_kspace_{}_error.png".format(config["config_index"], epoch + 1),
                        is_kspace=True)
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
                    "config_{}_recon_{}_{:.4g}_psnr_{:.4g}_ssim.png".format(config["config_index"],
                                                                            epoch + 1, test_psnr, test_ssim))
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

    return {"best_psnr": best_psnr, "best_psnr_ep": best_psnr_ep, "best_ssim": best_ssim, "best_ssim_ep": best_ssim_ep}
