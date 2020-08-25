from models.adain.AdaIN import AdaIN_Trainer
from data import ImgDataset, InfiniteSamplerWrapper
from options import hparams
from torch.utils.data import DataLoader
from torchvision import transforms
from loguru import logger
import numpy as np
import os, cv2
from PIL import Image
import torch

if __name__ == '__main__':
    trainer = AdaIN_Trainer(hparams)

    trans = transforms.Compose([
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor()
    ])
    content_dataset = ImgDataset(hparams.content_dir, trans)
    style_dataset = ImgDataset(hparams.style_dir, trans)

    content_dataloader = DataLoader(content_dataset, batch_size=hparams.batch_size,
                                    drop_last=True, sampler=InfiniteSamplerWrapper(content_dataset))
    style_dataloader = DataLoader(style_dataset, batch_size=hparams.batch_size,
                                  drop_last=True, sampler=InfiniteSamplerWrapper(style_dataset))
    content_dataloader = iter(content_dataloader)
    style_dataloader = iter(style_dataloader)

    alpha = 0.5
    if hparams.use_gpu:
        torch.backends.cudnn.deterministic = True
        device = torch.device("cuda")
        logger.info("CUDA visible devices: " + str(torch.cuda.device_count()))
        logger.info("CUDA Device Name: " + str(torch.cuda.get_device_name(device)))
        trainer.net = trainer.net.to(device)
        trainer.net = torch.nn.DataParallel(trainer.net)

    trainer.net.train()
    for epoch in range(hparams.epochs):

        contents = next(content_dataloader)
        styles = next(style_dataloader)

        if hparams.use_gpu:
            contents = contents.to(device)
            styles = styles.to(device)

        transfers, losses_ = trainer.train_step(contents, styles, alpha, global_step=epoch)
        loss_info = f'Step-{epoch}: '
        for key, item in losses_.items():
            loss_info += f'{key}: {item} | '
        logger.info(loss_info)

        if epoch % hparams.log_interval==0:
            content = contents[0].detach().cpu().numpy().transpose((1,2,0))
            style = styles[0].detach().cpu().numpy().transpose((1,2,0))
            tran = transfers[0].detach().cpu().numpy().transpose((1,2,0))
            content, tran, style = content[:, :, ::-1], tran[:, :, ::-1], style[:, :, ::-1]
            pano = np.concatenate([content, tran, style], axis=1)
            pano = np.uint8(pano * 255)
            file = os.path.join(f'{hparams.train_result_root}', f'{epoch}.png')
            cv2.imwrite(file, pano)

            trainer.net.eval().cpu()
            torch.save(trainer.net.state_dict(), hparams.weight_root+'/best.pth')
            if hparams.use_gpu:
                trainer.net.to(device).train()
            else:
                trainer.net.train()

