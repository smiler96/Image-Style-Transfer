from models.adain.AdaIN import AdaIN_Trainer
from data import ImgDataset, InfiniteSamplerWrapper
from options import hparams
from torch.utils.data import DataLoader
from torchvision import transforms
from loguru import logger
import numpy as np
import os, cv2, shutil
from PIL import Image
from argparse import ArgumentParser
import torch


def check_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)

def check_file(file):
    if os.path.exists(file):
        os.remove(file)

def check_paras(hparams):
    os.makedirs(hparams.save_root, exist_ok=True)
    test_result_root = os.path.join(hparams.save_root, 'test_results', hparams.model_name)
    os.makedirs(test_result_root, exist_ok=True)
    hparams.test_result_root = test_result_root 
    
    return hparams


if __name__ == '__main__':

    hparams = check_paras(hparams)
    
    log_file = os.path.join(hparams.save_root, 'loggers', hparams.model_name, 'test.txt') 
    check_file(log_file)
    logger.add(log_file, rotation="200 MB", backtrace=True, diagnose=True)
    logger.info(str(hparams))

    trainer = AdaIN_Trainer(hparams)

    if hparams.use_gpu:
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        logger.info("CUDA visible devices: " + str(torch.cuda.device_count()))
        logger.info("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))
        trainer.net = trainer.net.to(device)
        trainer.net = torch.nn.DataParallel(trainer.net)
        
    hparams.weight_root = os.path.join(hparams.save_root, 'weights', hparams.model_name)
    pretrained_dict = torch.load(hparams.weight_root + '/best.pth')
    
#     logger.info(str(pretrained_dict.keys()))
#     logger.info(str(trainer.net.state_dict().keys())) 
    trainer.net.load_state_dict(pretrained_dict)
    logger.info(f"Load model with {hparams.weight_root}/best.pth for testing.")

    trans = transforms.Compose([
#         transforms.Resize(size=(512, 512)),
#         transforms.RandomCrop(256),
        transforms.ToTensor(),
    ])

    trainer.net.eval()
    content_ = Image.open(hparams.content_file).convert('RGB')
    content_ = trans(content_)
    content_ = content_.unsqueeze(0)
    style_ = Image.open(hparams.style_file).convert('RGB')
    style_ = trans(style_)
    style_ = style_.unsqueeze(0)
    base_name_ = os.path.basename(hparams.content_file).split('.')[0] + "_" + \
                 os.path.basename(hparams.style_file).split('.')[0] + '_' + str(hparams.alpha) + '.png'
#     logger.info(content_)
#     logger.info(style_)
    if hparams.use_gpu:
        content_ = content_.to(device)
        style_ = style_.to(device)

    transfer_ = trainer.test_step(content_, style_, hparams.alpha)
    transfer_ = transfer_['transfer_content']
    transfer_ = transfer_[0].detach().cpu().numpy().transpose((1, 2, 0))
    transfer_ = np.uint8(transfer_ * 255)[:, :, ::-1]
    file = os.path.join(f'{hparams.test_result_root}', base_name_)
    cv2.imwrite(file, transfer_)

