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
from monai.data.utils import list_data_collate
from monai.transforms import (Compose, LoadImaged, Orientationd, ScaleIntensityd,
                              EnsureChannelFirstd, ToTensord, CropForegroundd, Spacingd, RandSpatialCropSamplesd,
                              Invertd, RandCropByPosNegLabeld, AsDiscreted, EnsureTyped, Activationsd, SpatialPadd)
from monai.handlers import (StatsHandler, TensorBoardStatsHandler, TensorBoardImageHandler,
                            MeanDice, CheckpointSaver, CheckpointLoader, SegmentationSaver,
                            ValidationHandler, LrScheduleHandler, HausdorffDistance, SurfaceDistance)
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

    keys = ("image", "label")
    train_tform = Compose([
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Spacingd(keys, 0.3, diagonal=True, mode=('bilinear', 'nearest')),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        CropForegroundd(keys, source_key="image"),
        SpatialPadd(keys, spatial_size=(96,96,96)),
        RandCropByPosNegLabeld(keys, label_key='label', spatial_size=(96,96,96), pos=0.8, neg=0.2, num_samples=2),
        EnsureTyped(keys),
        AsDiscreted(keys='label', to_onehot=True, n_classes=2)
    ])

    seg_tform = Compose([
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Spacingd(keys, 0.3, diagonal=True, mode='bilinear'),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        CropForegroundd(keys, source_key="image"),
        EnsureTyped(keys),
        AsDiscreted(keys='label', to_onehot=True, n_classes=2)
    ])

    post_tform = Compose(
        [Activationsd(keys='pred', softmax=True),
         AsDiscreted(keys='pred', argmax=True, to_onehot=True, n_classes=2),
         # AsDiscreted(keys=('label','pred'), to_onehot=True, n_classes=2),
        ]
    )

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = UNet(dimensions=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH, dropout=0.0).to(device)

    # model = UNETR(in_channels=1, out_channels=2, img_size=(96,96,96), feature_size=16, hidden_size=768, mlp_dim=3072,
    #               num_heads=12, pos_embed='perceptron', norm_name='instance', dropout_rate=0.2).to(device)

    @classmethod
    def load_train_data(cls, path, use_val=False):
        train_path = Path(path).joinpath('train')

        images = [str(p.absolute()) for p in train_path.glob("*US.nii")]
        segs = [str(p.absolute()) for p in train_path.glob("*label.nii")]

        if use_val:
            val_path = Path(path).joinpath('val')
            images += [str(p.absolute()) for p in val_path.glob("*US.nii")]
            segs += [str(p.absolute()) for p in val_path.glob("*label.nii")]

        images.sort()
        segs.sort()

        d = [{"image": im, "label": seg} for im, seg in zip(images, segs)]

        # ds = CacheDataset(d, xform)
        persistent_cache = Path("./persistent_cache")
        persistent_cache.mkdir(parents=True, exist_ok=True)
        ds = PersistentDataset(d, cls.train_tform, persistent_cache)
        loader = DataLoader(ds, batch_size=3, shuffle=True, num_workers=0, drop_last=True, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_val_data(cls, path, persistent=True, test=False):
        if not test:
            path = Path(path).joinpath('val')
        else:
            path = Path(path).joinpath('test')

        random.seed(0)
        images = sorted(str(p.absolute()) for p in path.glob("*US.nii"))
        segs = sorted(str(p.absolute()) for p in path.glob("*label.nii"))
        d = [{"image": im, "label": seg} for im, seg in zip(images, segs)]

        # ds = CacheDataset(d, xform)
        if persistent and not test:
            persistent_cache = Path("./persistent_cache")
            persistent_cache.mkdir(parents=True, exist_ok=True)
            ds = PersistentDataset(d, cls.seg_tform, persistent_cache)
        else:
            ds = Dataset(d, cls.seg_tform)
        loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, pin_memory=torch.cuda.is_available())

        return loader

    @classmethod
    def load_seg_data(cls, path):
        path = Path(path)

        random.seed(0)
        images = [str(p.absolute()) for p in path.glob("*.nii")]
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

        loss = DiceCELoss(softmax=True, include_background=False)

        opt = Novograd(net.parameters(), 1e-2)

        # trainer = create_supervised_trainer(net, opt, loss, device, False, )
        trainer = SupervisedTrainer(
            device=cls.device,
            max_epochs=2000,
            train_data_loader=loader,
            network=net,
            optimizer=opt,
            loss_function=loss,
            # decollate=False,
            key_train_metric={"train_meandice": MeanDice(output_transform=cls.output_tform)},
            amp=True
        )

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
            logdir = Path('./runs/')
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

        tb_writer = SummaryWriter(log_dir=logdir)
        tb_writer.add_graph(net, monai.utils.first(loader)['image'].to(cls.device))

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
            key_val_metric={"val_meandice": MeanDice(output_transform=cls.output_tform)},
            additional_metrics={
                'HausdorffDistance': HausdorffDistance(percentile=95, output_transform=cls.output_tform),
                'AvgSurfaceDistance': SurfaceDistance(output_transform=cls.output_tform)},
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
            loader = cls.load_val_data("U:/Documents/DeepMV/data", test=use_test)
        else:
            loader = cls.load_val_data(data, False, test=use_test)

        net = cls.load_model(load_checkpoint)

        evaluator = SupervisedEvaluator(
            device=cls.device,
            val_data_loader=loader,
            network=net,
            inferer=SlidingWindowInferer((96, 96, 96), sw_batch_size=6),
            key_val_metric={"val_meandice": MeanDice(output_transform=cls.output_tform)},
            additional_metrics={
                'HausdorffDistance': HausdorffDistance(percentile=95, include_background=True, output_transform=cls.output_tform),
                'AvgSurfaceDistance': SurfaceDistance(include_background=True, output_transform=cls.output_tform)},
        )

        # checkpoint_loader = CheckpointLoader(load_checkpoint, {'net': net})
        # checkpoint_loader.attach(evaluator)

        logdir = Path(load_checkpoint).parent.joinpath('out')

        # add stats event handler to print validation stats via evaluator
        val_stats_handler = StatsHandler(
            name="evaluator",
            output_transform=lambda x: None,
            global_epoch_transform=lambda x: 0)
        val_stats_handler.attach(evaluator)

        def tx(output):
            return predict_segmentation(cls.output_tform(output)[0], mutually_exclusive=True)

        prediction_saver = SegmentationSaver(
            output_dir=logdir,
            name="evaluator",
            dtype=np.dtype('float64'),
            batch_transform=lambda batch: list_data_collate(batch)["image_meta_dict"],
            output_transform=tx
        )
        prediction_saver.attach(evaluator)

        start = timer()
        evaluator.run()
        end = timer()
        print(evaluator.get_validation_stats(), "took {}s".format(end-start))

    @classmethod
    def segment(cls, load_checkpoint, data):
        config.print_config()

        loader = cls.load_seg_data(data)

        device = torch.device('cuda:0')
        net = cls.load_model(load_checkpoint)

        evaluator = SupervisedEvaluator(
            device=device,
            val_data_loader=loader,
            network=net,
            decollate=False,
            inferer=SlidingWindowInferer((96,96,96), sw_batch_size=6),
        )

        logdir = Path(load_checkpoint).parent.joinpath('out')

        prediction_saver = SegmentationSaver(
            output_dir=logdir,
            name="evaluator",
            dtype=np.dtype('float64'),
            batch_transform=lambda batch: batch["image_meta_dict"],
            output_transform=lambda output: predict_segmentation(output['pred'])
        )
        prediction_saver.attach(evaluator)

        evaluator.run()

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