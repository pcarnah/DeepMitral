from pathlib import Path

import numpy as np
import logging
import sys
import torch
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator

import monai
from monai import config
from monai.data import Dataset, DataLoader, GridPatchDataset, CacheDataset
from monai.transforms import (Compose, LoadNiftid, Orientationd, ScaleIntensityd,
                              AddChanneld, ToTensord, CropForegroundd, Spacingd, RandSpatialCropSamplesd, AsDiscreted)
from monai.handlers import (StatsHandler, TensorBoardStatsHandler, TensorBoardImageHandler,
                            MeanDice, CheckpointSaver)
from monai.networks import predict_segmentation
from monai.networks.nets import UNet
from monai.networks.layers import Norm
from monai.losses import DiceLoss
from monai.inferers import SlidingWindowInferer
from monai.engines import SupervisedTrainer


def main():
    config.print_config()

    path = Path("D:/pcarnahanfiles/Tensorflow/MVData/nifty")

    images = sorted(str(p.absolute()) for p in path.glob("*US.nii"))
    segs = sorted(str(p.absolute()) for p in path.glob("*label.nii"))
    d = [{"image": im, "label": seg} for im, seg in zip(images, segs)]
    keys = ("image", "label")

    # Define transforms for image and segmentation
    xform = Compose([
        LoadNiftid(keys),
        AddChanneld(keys),
        Spacingd(keys, 0.7, diagonal=True, mode=('bilinear', 'nearest')),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        CropForegroundd(keys, source_key="image"),
        RandSpatialCropSamplesd(keys, (96,96,96), 8, random_size=False),
        ToTensord(keys)
    ])
    # imtrans = Compose([
    #     LoadNifti(image_only=True),
    #     ScaleIntensity(),
    #     AddChannel(),
    #     ToTensor()
    # ])
    # segtrans = Compose([
    #     LoadNifti(image_only=True),
    #     AddChannel(),
    #     ToTensor()
    # ])

    # ds = ArrayDataset(images, imtrans, segs, segtrans)
    # ds = Dataset(d, xform)
    ds = CacheDataset(d, xform)
    loader = DataLoader(ds,  batch_size=4, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())
    im = monai.utils.misc.first(loader)
    print(im["image"].shape)

    device = torch.device('cuda:0')
    net = UNet(dimensions=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256),
               strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)
    loss = DiceLoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-2)

    # trainer = create_supervised_trainer(net, opt, loss, device, False, )
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=500,
        train_data_loader=loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        # inferer=SlidingWindowInferer((64, 64, 64), sw_batch_size=6),
        # post_transform=AsDiscreted(keys=["pred", "label"], argmax=(True, False), to_onehot=False, n_classes=2),
        key_train_metric={"train_meandice": MeanDice(output_transform=lambda x: (x["pred"], x["label"]))},
    )

    ### optional section for checkpoint and tensorboard logging
    # adding checkpoint handler to save models (network params and optimizer stats) during training
    logdir = Path('./runs/')
    dirs = sorted([x.name for x in logdir.iterdir() if x.is_dir()])
    if not dirs:
        run = 0
    else:
        run = int(dirs[-1]) + 1

    checkpoint_handler = CheckpointSaver('./runs/{}'.format(run), {'net': net, 'opt': opt}, n_saved=10, save_final=True,
                                         save_key_metric=True, key_metric_n_saved=5, epoch_level=True, save_interval=2)
    checkpoint_handler.attach(trainer)

    # StatsHandler prints loss at every iteration and print metrics at every epoch,
    # we don't set metrics for trainer here, so just print loss, user can also customize print functions
    # and can use output_transform to convert engine.state.output if it's not a loss value
    train_stats_handler = StatsHandler(name='trainer')
    train_stats_handler.attach(trainer)

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler(log_dir='./runs/{}'.format(run))
    train_tensorboard_stats_handler.attach(trainer)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    trainer.run()


if __name__ == "__main__":
    main()