from pathlib import Path

import numpy as np
import logging
import sys
import torch
import argparse
import random

import monai
from monai import config
from monai.data import Dataset, DataLoader, GridPatchDataset, CacheDataset, PersistentDataset
from monai.transforms import (Compose, LoadNiftid, Orientationd, ScaleIntensityd,
                              AddChanneld, ToTensord, CropForegroundd, Spacingd, RandSpatialCropSamplesd,
                              RandAffined, RandCropByPosNegLabeld, AsDiscreted, Rand3DElasticd)
from monai.handlers import (StatsHandler, TensorBoardStatsHandler, TensorBoardImageHandler,
                            MeanDice, CheckpointSaver, CheckpointLoader, SegmentationSaver,
                            ValidationHandler, LrScheduleHandler)
from monai.networks import predict_segmentation
from monai.networks.nets import *
from monai.networks.layers import Norm
from monai.losses import DiceLoss, GeneralizedDiceLoss
from monai.inferers import SlidingWindowInferer
from monai.engines import SupervisedTrainer, SupervisedEvaluator

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau

def parse_args():
    parser = argparse.ArgumentParser(description='DeepMV training')

    subparsers = parser.add_subparsers(help='sub-command help', dest='mode')
    subparsers.required = True

    train_parse = subparsers.add_parser('train', help="Train the network")
    train_parse.add_argument('-load', type=str, help='load from a given checkpoint')
    train_parse.add_argument('-data', type=str, help='data folder')

    val_parse = subparsers.add_parser('validate', help='Evaluate the network')
    val_parse.add_argument('load', type=str, help='load from a given checkpoint')
    val_parse.add_argument('-data', type=str, help='data folder')

    seg_parse = subparsers.add_parser('segment', help='Segment images')
    seg_parse.add_argument('load', type=str, help='load from a given checkpoint')
    seg_parse.add_argument('data', type=str, help='data folder')

    return parser.parse_args()

def load_train_data(path, device=torch.device('cpu')):
    path = Path(path)

    images = sorted(str(p.absolute()) for p in path.glob("*US.nii"))
    segs = sorted(str(p.absolute()) for p in path.glob("*label.nii"))
    d = [{"image": im, "label": seg} for im, seg in zip(images, segs)]
    keys = ("image", "label")

    # Define transforms for image and segmentation
    xform = Compose([
        LoadNiftid(keys),
        AddChanneld(keys),
        Spacingd(keys, 0.5, diagonal=True, mode=('bilinear', 'nearest')),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        CropForegroundd(keys, source_key="image"),
        #RandAffined(keys, mode=('bilinear', 'nearest'), rotate_range=(0.15,0.15,0.15), scale_range=(0.05,0.05,0.05), prob=0.2, as_tensor_output=False, device=device),
        # Rand3DElasticd(keys, mode=('bilinear', 'nearest'),  rotate_range=(0.15,0.15,0.15), scale_range=(0.05,0.05,0.05),
        #                sigma_range=(0,1), magnitude_range=(0,2), prob=0.2, as_tensor_output=False, device=device),
        RandCropByPosNegLabeld(keys, label_key='label', spatial_size=(96,96,96), pos=0.8, neg=0.2, num_samples=4),
        ToTensord(keys)
    ])

    # ds = CacheDataset(d, xform)
    persistent_cache = Path("./persistent_cache")
    persistent_cache.mkdir(parents=True, exist_ok=True)
    ds = PersistentDataset(d, xform, persistent_cache)
    loader = DataLoader(ds, batch_size=7, shuffle=True, num_workers=0, drop_last=True)

    return loader

def load_val_data(path, persistent=True):
    path = Path(path)

    random.seed(0)
    images = sorted(str(p.absolute()) for p in path.glob("*US.nii"))
    segs = sorted(str(p.absolute()) for p in path.glob("*label.nii"))
    d = [{"image": im, "label": seg} for im, seg in zip(images, segs)]
    keys = ("image", "label")

    # Define transforms for image and segmentation
    xform = Compose([
        LoadNiftid(keys),
        AddChanneld(keys),
        Spacingd(keys, 0.5, diagonal=True, mode=('bilinear', 'nearest')),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        CropForegroundd(keys, source_key="image"),
        ToTensord(keys)
    ])

    # ds = CacheDataset(d, xform)
    if persistent:
        persistent_cache = Path("./persistent_cache")
        persistent_cache.mkdir(parents=True, exist_ok=True)
        ds = PersistentDataset(d, xform, persistent_cache)
    else:
        ds = Dataset(d, xform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    return loader

def load_seg_data(path):
    path = Path(path)

    random.seed(0)
    images = [str(p.absolute()) for p in path.glob("*.nii")]
    d = [{"image": im} for im in images]
    keys = ("image")

    # Define transforms for image and segmentation
    xform = Compose([
        LoadNiftid(keys),
        AddChanneld(keys),
        Spacingd(keys, 0.5, diagonal=True, mode='bilinear'),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        CropForegroundd(keys, source_key="image"),
        ToTensord(keys)
    ])

    # ds = CacheDataset(d, xform)
    ds = Dataset(d, xform)
    loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

    return loader


def train(args):
    config.print_config()

    device = torch.device('cuda:0')

    if not args.data:
        loader = load_train_data("U:/Documents/DeepMV/data/train", device)
    else:
        loader = load_train_data(args.data, device)

    net = UNet(dimensions=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256),
               strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH, dropout=0.05).to(device)
    loss = GeneralizedDiceLoss(sigmoid=True)
    opt = torch.optim.Adam(net.parameters(), 1e-3)

    # trainer = create_supervised_trainer(net, opt, loss, device, False, )
    trainer = SupervisedTrainer(
        device=device,
        max_epochs=1200,
        train_data_loader=loader,
        network=net,
        optimizer=opt,
        loss_function=loss,
        key_train_metric={"train_meandice": MeanDice(sigmoid=True,output_transform=lambda x: (x["pred"], x["label"]))},
    )

    # Load checkpoint if defined
    if args.load:
        checkpoint = torch.load(args.load)
        if checkpoint['trainer']:
            for k in checkpoint['trainer']:
                trainer.state.__dict__[k] = checkpoint['trainer'][k]
            trainer.state.epoch = trainer.state.iteration // trainer.state.epoch_length
        checkpoint_loader = CheckpointLoader(args.load, {'net': net, 'opt': opt})
        checkpoint_loader.attach(trainer)

        logdir = Path(args.load).parent

    else:
        logdir = Path('./runs/')
        dirs = sorted([int(x.name) for x in logdir.iterdir() if x.is_dir()])
        if not dirs:
            logdir = logdir.joinpath('0')
        else:
            logdir = logdir.joinpath(str(int(dirs[-1]) + 1))

    # Adaptive learning rate
    lr_scheduler = StepLR(opt, 600)
    lr_handler = LrScheduleHandler(lr_scheduler)
    lr_handler.attach(trainer)

    ### optional section for checkpoint and tensorboard logging
    # adding checkpoint handler to save models (network params and optimizer stats) during training
    checkpoint_handler = CheckpointSaver(logdir, {'net': net, 'opt': opt, 'trainer': trainer}, n_saved=20, save_final=True,
                                         epoch_level=True, save_interval=50)
    checkpoint_handler.attach(trainer)

    # StatsHandler prints loss at every iteration and print metrics at every epoch
    train_stats_handler = StatsHandler(
        name='trainer',
        output_transform=lambda x: x['loss'])
    train_stats_handler.attach(trainer)

    tb_writer = SummaryWriter(log_dir=logdir)
    tb_writer.add_graph(net, monai.utils.first(loader)['image'].to(device))

    # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
    train_tensorboard_stats_handler = TensorBoardStatsHandler(
        summary_writer=tb_writer,
        output_transform=lambda x: x['loss'],
        tag_name='loss')
    train_tensorboard_stats_handler.attach(trainer)

    # Set up validation step
    val_loader = load_val_data("U:/Documents/DeepMV/data/val")

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=net,
        inferer=SlidingWindowInferer((96, 96, 96), sw_batch_size=6),
        key_val_metric={"val_meandice": MeanDice(sigmoid=True, output_transform=lambda x: (x["pred"], x["label"]))},
    )

    val_stats_handler = StatsHandler(
        name='evaluator',
        output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch)  # fetch global epoch number from trainer
    val_stats_handler.attach(evaluator)

    # add handler to record metrics to TensorBoard at every validation epoch
    val_tensorboard_stats_handler = TensorBoardStatsHandler(
        summary_writer=tb_writer,
        output_transform=lambda x: None,  # no need to plot loss value, so disable per iteration output
        global_epoch_transform=lambda x: trainer.state.epoch)  # fetch global epoch number from trainer
    val_tensorboard_stats_handler.attach(evaluator)

    # add handler to draw the first image and the corresponding label and model output in the last batch
    # here we draw the 3D output as GIF format along Depth axis, at every validation epoch
    val_tensorboard_image_handler = TensorBoardImageHandler(
        summary_writer=tb_writer,
        batch_transform=lambda batch: (batch["image"], batch["label"]),
        output_transform=lambda output: predict_segmentation(output['pred']),
        global_iter_transform=lambda x: trainer.state.epoch
    )
    val_tensorboard_image_handler.attach(evaluator)

    val_handler = ValidationHandler(
        validator=evaluator,
        interval=10
    )
    val_handler.attach(trainer)

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    trainer.run()


def validate(args):
    config.print_config()

    if not args.data:
        loader = load_val_data("U:/Documents/DeepMV/data/val")
    else:
        loader = load_val_data(args.data, False)

    device = torch.device('cuda:0')
    net = UNet(dimensions=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256),
               strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=loader,
        network=net,
        inferer=SlidingWindowInferer((96,96,96), sw_batch_size=6),
        key_val_metric={"val_meandice": MeanDice(sigmoid=True,output_transform=lambda x: (x["pred"], x["label"]))},
    )

    checkpoint_loader = CheckpointLoader(args.load, {'net': net})
    checkpoint_loader.attach(evaluator)

    logdir = Path(args.load).parent.joinpath('out')

    # add stats event handler to print validation stats via evaluator
    val_stats_handler = StatsHandler(
        name="evaluator",
        output_transform=lambda x: None,
        global_epoch_transform=lambda x: 0)
    val_stats_handler.attach(evaluator)

    prediction_saver = SegmentationSaver(
        output_dir=logdir,
        name="evaluator",
        batch_transform=lambda batch: batch["image_meta_dict"],
        output_transform=lambda output: predict_segmentation(output['pred'])
    )
    prediction_saver.attach(evaluator)

    evaluator.run()
    print(evaluator.get_validation_stats())

def segment(args):
    config.print_config()

    loader = load_seg_data(args.data)

    device = torch.device('cuda:0')
    net = UNet(dimensions=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128, 256),
               strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH).to(device)

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=loader,
        network=net,
        inferer=SlidingWindowInferer((96,96,96), sw_batch_size=6),
    )

    checkpoint_loader = CheckpointLoader(args.load, {'net': net})
    checkpoint_loader.attach(evaluator)

    logdir = Path(args.load).parent.joinpath('out')

    prediction_saver = SegmentationSaver(
        output_dir=logdir,
        name="evaluator",
        batch_transform=lambda batch: batch["image_meta_dict"],
        output_transform=lambda output: predict_segmentation(output['pred'])
    )
    prediction_saver.attach(evaluator)

    evaluator.run()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == 'validate':
        validate(args)
    elif args.mode == 'train':
        train(args)
    elif args.mode == 'segment':
        segment(args)