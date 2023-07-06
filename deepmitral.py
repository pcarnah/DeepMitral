import os
import platform
from pathlib import Path
from typing import Tuple

import numpy as np
import logging
import sys
import torch
import argparse
import random

import monai
from ignite.metrics import Average
from monai import config
from monai.data import Dataset, DataLoader, GridPatchDataset, CacheDataset, PersistentDataset, LMDBDataset
from monai.data.utils import list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.transforms import (Compose, LoadImaged, Orientationd, ScaleIntensityd,
                              EnsureChannelFirstd, ToTensord, CropForegroundd, Spacingd, RandSpatialCropSamplesd,
                              Invertd, RandCropByPosNegLabeld, AsDiscreted, EnsureTyped, Activationsd, SpatialPadd, DataStatsd,
                              KeepLargestConnectedComponentd, SaveImage, AsDiscrete)
from monai.handlers import (StatsHandler, TensorBoardStatsHandler, TensorBoardImageHandler,
                            MeanDice, CheckpointSaver, CheckpointLoader,
                            ValidationHandler, LrScheduleHandler, HausdorffDistance, SurfaceDistance)
from monai.metrics import GeneralizedDiceScore, HausdorffDistanceMetric, SurfaceDistanceMetric
from monai.networks import predict_segmentation
from monai.networks.nets import *
from monai.networks.layers import Norm
from monai.optimizers import Novograd
from monai.losses import DiceLoss, GeneralizedDiceLoss, DiceCELoss
from monai.inferers import SlidingWindowInferer
from monai.engines import SupervisedTrainer, SupervisedEvaluator

from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau


from timeit import default_timer as timer

class DeepMitral:

    spacing = 0.3
    train_epochs = 500

    keys = ("image", "label")
    train_tform = Compose([
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Spacingd(keys, (spacing, spacing, spacing), diagonal=True, mode=('bilinear', 'nearest')),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        CropForegroundd(keys, source_key="image"),
        SpatialPadd(keys, spatial_size=(96,96,96)),
        RandCropByPosNegLabeld(keys, label_key='label', spatial_size=(96,96,96), pos=0.8, neg=0.2, num_samples=8),
        EnsureTyped(keys),
        AsDiscreted(keys='label', to_onehot=2)
    ])

    val_tform = Compose([
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Spacingd(keys, (spacing, spacing, spacing), diagonal=True, mode=('bilinear', 'nearest')),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        EnsureTyped(keys),
        AsDiscreted(keys='label', to_onehot=2)
    ])

    seg_tform = Compose([
        LoadImaged('image'),
        EnsureChannelFirstd('image'),
        Spacingd('image', (spacing, spacing, spacing), diagonal=True, mode='bilinear'),
        Orientationd('image', axcodes='RAS'),
        ScaleIntensityd("image"),
        EnsureTyped('image'),
    ])

    post_tform = Compose(
        [Activationsd(keys='pred', softmax=True),
         AsDiscreted(keys='pred', argmax=True, to_onehot=2),
         KeepLargestConnectedComponentd(keys='pred', applied_labels=1)
         # AsDiscreted(keys=('label','pred'), to_onehot=True, n_classes=2),
        ]
    )

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH, dropout=0).to(device)

    # model = FlexibleUNet(spatial_dims=3, in_channels=1, out_channels=2, backbone='efficientnet-b0', act='memswish').to(device)

    # model = UNETR(in_channels=1, out_channels=2, img_size=(96,96,96), feature_size=16, hidden_size=768, mlp_dim=3072,
    #               num_heads=12, pos_embed='perceptron', norm_name='instance', dropout_rate=0.2).to(device)

    # model = SwinUNETR((96, 96, 96), 1, 2, depths=(2, 4, 2, 2), feature_size=12).to(device)

    trainer = None

    @classmethod
    def load_train_data(cls, path, use_val=False):
        train_path = Path(path).joinpath('train')

        images = [str(p.absolute()) for p in train_path.glob("*US.nii*")]
        segs = [str(p.absolute()) for p in train_path.glob("*label.nii*")]

        if use_val:
            val_path = Path(path).joinpath('val')
            images += [str(p.absolute()) for p in val_path.glob("*US.nii*")]
            segs += [str(p.absolute()) for p in val_path.glob("*label.nii*")]

        images.sort()
        segs.sort()

        d = [{"image": im, "label": seg} for im, seg in zip(images, segs)]

        # if platform.system() == 'Windows':
        #     persistent_cache = Path("./persistent_cache")
        #     persistent_cache.mkdir(parents=True, exist_ok=True)
        #     ds = LMDBDataset(d, cls.train_tform, cache_dir=persistent_cache,
        #                      lmdb_kwargs={'writemap': True, 'map_size': 100000000})
        # else:
        #     num_workers = os.cpu_count()
        ds = CacheDataset(d, cls.train_tform)
        loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=0, drop_last=True, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_val_data(cls, path, persistent=True, test=False):
        if not test:
            path = Path(path).joinpath('val')
        else:
            path = Path(path).joinpath('test')

        random.seed(0)
        images = sorted(str(p.absolute()) for p in path.glob("*US.nii*"))
        segs = sorted(str(p.absolute()) for p in path.glob("*label.nii*"))
        d = [{"image": im, "label": seg} for im, seg in zip(images, segs)]

        # ds = CacheDataset(d, xform)
        if persistent and not test:
            persistent_cache = Path("./persistent_cache")
            persistent_cache.mkdir(parents=True, exist_ok=True)
            ds = PersistentDataset(d, cls.val_tform, persistent_cache)
        else:
            ds = Dataset(d, cls.val_tform)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_seg_data(cls, path):
        path = Path(path)

        random.seed(0)
        images = [str(p.absolute()) for p in path.glob("*.nii*")]
        d = [{"image": im} for im in images]

        # ds = CacheDataset(d, xform)
        ds = Dataset(d, cls.seg_tform)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_model(cls, path):
        net_params = torch.load(path, map_location=cls.device)

        if isinstance(net_params, dict):
            cls.model.load_state_dict(net_params['net'])
        elif isinstance(net_params, torch.nn.Module):
            cls.model = net_params
        else:
            logging.error('Could not load network')
            return

        return cls.model

    @classmethod
    def output_tform(cls, x):
        xt = list_data_collate(cls.post_tform(x))
        return xt['pred'], xt['label']

    @classmethod
    def train(cls, data=None, use_val=False, load_checkpoint=None):
        config.print_config()

        if not data:
            dataPath = "U:/Documents/DeepMV/data/"
        else:
            dataPath = data

        loader = cls.load_train_data(dataPath, use_val=use_val)

        net = cls.model

        # loss = GeneralizedDiceLoss(sigmoid=True)
        # loss = SDWeightedDiceLoss(sigmoid=True)
        # opt = torch.optim.Adam(net.parameters(), 1e-3)

        # gd = GeneralizedDiceLoss(sigmoid=True)
        # bce = torch.nn.BCEWithLogitsLoss()
        #
        # def loss(input,target):
        #     return gd(input,target) + bce(input,target)

        loss = DiceCELoss(softmax=True, include_background=False, lambda_dice=0.5)

        opt = Novograd(net.parameters(), 1e-3)

        # trainer = create_supervised_trainer(net, opt, loss, device, False, )
        trainer = SupervisedTrainer(
            device=cls.device,
            max_epochs=cls.train_epochs,
            train_data_loader=loader,
            network=net,
            optimizer=opt,
            loss_function=loss,
            # decollate=False,
            key_train_metric={"train_meandice": MeanDice(output_transform=cls.output_tform)},
            amp=True
        )
        cls.trainer = trainer

        # Load checkpoint if defined
        if load_checkpoint:
            checkpoint = torch.load(load_checkpoint)
            if checkpoint['trainer']:
                for k in checkpoint['trainer']:
                    trainer.state.__dict__[k] = checkpoint['trainer'][k]
                trainer.state.epoch = trainer.state.iteration // trainer.state.epoch_length
            checkpoint_loader = CheckpointLoader(load_checkpoint, {'net': net, 'opt': opt})
            checkpoint_loader.attach(trainer)

            logdir = Path(load_checkpoint).parent

        else:
            logdir = Path(dataPath).joinpath('runs')
            logdir.mkdir(exist_ok=True)
            dirs = sorted([int(x.name) for x in logdir.iterdir() if x.is_dir()])
            if not dirs:
                logdir = logdir.joinpath('0')
            else:
                logdir = logdir.joinpath(str(int(dirs[-1]) + 1))

        # Adaptive learning rate
        lr_scheduler = StepLR(opt, 1000)
        lr_handler = LrScheduleHandler(lr_scheduler)
        lr_handler.attach(trainer)

        ### optional section for checkpoint and tensorboard logging
        # adding checkpoint handler to save models (network params and optimizer stats) during training
        checkpoint_handler = CheckpointSaver(logdir, {'net': net, 'opt': opt, 'trainer': trainer}, n_saved=10, save_final=True,
                                             epoch_level=True, save_interval=200)
        checkpoint_handler.attach(trainer)

        # StatsHandler prints loss at every iteration and print metrics at every epoch
        train_stats_handler = StatsHandler(
            name='trainer',
            output_transform=lambda x: x[0]['loss'])
        train_stats_handler.attach(trainer)

        test = monai.utils.first(loader)['image']

        tb_writer = SummaryWriter(log_dir=str(logdir))
        # tb_writer.add_graph(net, monai.utils.first(loader)['image'].to(cls.device).as_tensor())

        # TensorBoardStatsHandler plots loss at every iteration and plots metrics at every epoch, same as StatsHandler
        train_tensorboard_stats_handler = TensorBoardStatsHandler(
            summary_writer=tb_writer,
            output_transform=lambda x: x[0]['loss'],
            tag_name='loss')
        train_tensorboard_stats_handler.attach(trainer)

        # Set up validation step
        val_loader = cls.load_val_data(dataPath)

        evaluator = SupervisedEvaluator(
            device=cls.device,
            val_data_loader=val_loader,
            network=net,
            inferer=SlidingWindowInferer((96, 96, 96), sw_batch_size=6),
            # decollate=False,
            key_val_metric={"val_meandice": MeanDice(output_transform=cls.output_tform, include_background=False)},
            additional_metrics={
                'HausdorffDistance': HausdorffDistance(percentile=95, output_transform=cls.output_tform, include_background=False),
                'AvgSurfaceDistance': SurfaceDistance(output_transform=cls.output_tform, include_background=False)},
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
            global_epoch_transform=lambda x: trainer.state.epoch,)  # fetch global epoch number from trainer
        val_tensorboard_stats_handler.attach(evaluator)

        # add handler to draw the first image and the corresponding label and model output in the last batch
        # here we draw the 3D output as GIF format along Depth axis, at every validation epoch
        # val_tensorboard_image_handler = TensorBoardImageHandler(
        #     summary_writer=tb_writer,
        #     batch_transform=lambda batch: (list_data_collate(batch)["image"], list_data_collate(batch)["label"]),
        #     output_transform=lambda output: predict_segmentation(list_data_collate(output)['pred'], mutually_exclusive=True),
        #     global_iter_transform=lambda x: trainer.state.epoch
        # )
        # val_tensorboard_image_handler.attach(evaluator)

        val_handler = ValidationHandler(
            validator=evaluator,
            interval=10
        )
        val_handler.attach(trainer)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        trainer.run()

        torch.save(net, logdir.joinpath('final_model.md'))

    @classmethod
    def validate(cls, load_checkpoint, data=None, use_test=False):
        config.print_config()

        if not data:
            data = "U:/Documents/DeepMV/data"
            loader = cls.load_val_data(data, test=use_test)
        else:
            loader = cls.load_val_data(data, False, test=use_test)

        net = cls.load_model(load_checkpoint)
        device = cls.device

        logdir = Path(load_checkpoint).parent.joinpath('out')

        pred = AsDiscrete(argmax=True)

        saver = SaveImage(
            output_dir=str(logdir),
            output_postfix="seg",
            output_ext=".nii.gz",
            output_dtype=np.uint8,
        )

        mean_dice = GeneralizedDiceScore()
        hd = HausdorffDistanceMetric(percentile=95, include_background=False)
        sd = SurfaceDistanceMetric(include_background=False)

        start = timer()
        net.eval()

        frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(net))

        with torch.no_grad():
            for batch in loader:
                label = batch['label'].to(device)
                out = sliding_window_inference(batch['image'].to(device), (96, 96, 96), 8, frozen_mod)
                out = cls.post_tform(decollate_batch({'pred': out}))

                mean_dice(list_data_collate(out)['pred'].cpu(), label.cpu())
                hd(list_data_collate(out)['pred'].cpu(), label.cpu())
                sd(list_data_collate(out)['pred'].cpu(), label.cpu())

                meta_dict = decollate_batch(batch["image_meta_dict"])
                for o, m in zip(out, meta_dict):
                    saver(pred(o['pred']), m)

        end = timer()

        print("Metric Mean_Dice: {}".format(mean_dice.aggregate().item()))
        print("Metric Mean_HD: {}".format(hd.aggregate().item() * cls.spacing))
        print("Metric Mean_SD: {}".format(sd.aggregate().item() * cls.spacing))
        print("Elapsed Time: {}s".format(end - start))


    @classmethod
    def segment(cls, load_checkpoint, data):
        config.print_config()

        loader = cls.load_seg_data(data)

        device = cls.device
        net = cls.load_model(load_checkpoint)

        logdir = Path(data).joinpath('out')

        pred = AsDiscrete(argmax=True)

        saver = SaveImage(
            output_dir=str(logdir),
            output_postfix="seg",
            output_ext=".nii.gz",
            output_dtype=np.uint8,
        )

        net.eval()
        with torch.no_grad():
            for batch in loader:
                with torch.cuda.amp.autocast():
                    out = sliding_window_inference(batch['image'].to(device), (96,96,96), 16, net)
                out = cls.post_tform(decollate_batch({'pred': out}))
                meta_dict = decollate_batch(batch["image_meta_dict"])
                for o, m in zip(out, meta_dict):
                    saver(pred(o['pred']), m)

    @classmethod
    def handle_sigint(cls, signum, frame):
        if cls.trainer:
            msg = "Ctrl-c was pressed. Stopping run at epoch {}.".format(cls.trainer.state.epoch)
            cls.trainer.should_terminate = True
            cls.trainer.should_terminate_single_epoch = True
        else:
            msg = "Ctrl-c was pressed. Stopping run."
            print(msg, flush=True)
            exit(1)
        print(msg, flush=True)



def parse_args():
    parser = argparse.ArgumentParser(description='DeepMV training')

    subparsers = parser.add_subparsers(help='sub-command help', dest='mode')
    subparsers.required = True

    train_parse = subparsers.add_parser('train', help="Train the network")
    train_parse.add_argument('-load', type=str, help='load from a given checkpoint')
    train_parse.add_argument('-data', type=str, help='data folder. should contain "train" and "val" sub-folders')
    train_parse.add_argument('-use_val', action='store_true', help='Flag to indicate that training set should include validation data.')

    val_parse = subparsers.add_parser('validate', help='Evaluate the network')
    val_parse.add_argument('load', type=str, help='load from a given checkpoint')
    val_parse.add_argument('-data', type=str, help='data folder. should contain "train" and "val" sub-folders')
    val_parse.add_argument('-use_test', action='store_true',
                             help='Run on test data')

    seg_parse = subparsers.add_parser('segment', help='Segment images')
    seg_parse.add_argument('load', type=str, help='load from a given checkpoint')
    seg_parse.add_argument('data', type=str, help='data folder')

    return parser.parse_args()

def main():
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    start = timer()
    args = parse_args()
    if args.mode == 'validate':
        DeepMitral.validate(args.load, args.data, args.use_test)
    elif args.mode == 'train':
        DeepMitral.train(args.data, args.use_val, args.load)
    elif args.mode == 'segment':
        DeepMitral.segment(args.load, args.data)
    end = timer()
    print({"Total runtime: {}".format(end - start)})


if __name__ == "__main__":
    main()