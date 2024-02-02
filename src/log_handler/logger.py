"""

A class to handle important scalars during training

"""
import torch.utils.tensorboard as tensorboardX

class INRLogger():
    def __init__(self, log_dir):
        self.writer = tensorboardX.SummaryWriter(log_dir=log_dir)

    def log_train(self, loss, epoch):
        self.writer.add_scalar("train_loss", loss, epoch)
    
    def log_test(self, loss, psnr, ssim, epoch):
        self.writer.add_scalar('test_loss', loss, epoch)
        self.writer.add_scalar('test_psnr', psnr, epoch)
        self.writer.add_scalar('test_ssim', ssim, epoch)