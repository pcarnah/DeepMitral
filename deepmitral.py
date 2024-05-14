import argparse
import logging
import os
import platform
import random
import shutil
import signal
import sys
from pathlib import Path
from timeit import default_timer as timer

import numpy as np
import torch
from monai import config
from monai.data import (Dataset, DataLoader, PersistentDataset, LMDBDataset,
                        CacheDataset, ThreadDataLoader, set_track_meta)
from monai.data.utils import list_data_collate, decollate_batch
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.engines.workflow import Events
from monai.handlers import (StatsHandler, TensorBoardStatsHandler, MeanDice,
                            CheckpointSaver, CheckpointLoader,
                            ValidationHandler, LrScheduleHandler,
                            HausdorffDistance, SurfaceDistance, GarbageCollector)
from monai.inferers import SlidingWindowInferer
from monai.inferers import sliding_window_inference
from monai.losses import DiceCELoss, GeneralizedDiceFocalLoss
from monai.metrics import (GeneralizedDiceScore, HausdorffDistanceMetric,
                           SurfaceDistanceMetric)
from monai.networks.layers import Norm
from monai.networks.nets import UNet, FlexibleUNet
from monai.optimizers import Novograd
from monai.transforms import (Compose, LoadImaged, Orientationd,
                              ScaleIntensityd,
                              EnsureChannelFirstd, CropForegroundd, Spacingd,
                              RandCropByPosNegLabeld, AsDiscreted,
                              EnsureTyped, Activationsd, SpatialPadd,
                              KeepLargestConnectedComponentd, SaveImage,
                              AsDiscrete, CenterSpatialCropd,
                              ClassesToIndicesd, RandCropByLabelClassesd)
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR
from torch.utils.tensorboard import SummaryWriter
import pandas as pd


class DeepMitral:
    spacing = 0.3
    train_epochs = 800
    n_classes = 2
    batch_size = 12
    n_samples = 4
    patch_size = (128,128,128)

    keys = ("image", "label")
    train_tform = Compose([
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Spacingd(keys, (spacing, spacing, spacing),
                 mode=('bilinear', 'nearest')),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        CropForegroundd(keys, source_key="image"),
        SpatialPadd(keys, spatial_size=patch_size),
        AsDiscreted(keys='label', to_onehot=n_classes),
        ClassesToIndicesd('label', image_key='image', image_threshold=0.05),
        RandCropByLabelClassesd(keys, label_key='label', num_samples=n_samples,
                                ratios=[0.5] + [1] * (n_classes - 1),
                                indices_key='label_cls_indices',
                                spatial_size=patch_size),
        # RandCropByPosNegLabeld(keys, label_key='label',
        #                        spatial_size=(96, 96, 96), pos=0.8, neg=0.2,
        #                        num_samples=4),
        CenterSpatialCropd(keys, patch_size),
        EnsureTyped(keys),
    ])

    val_tform = Compose([
        LoadImaged(keys),
        EnsureChannelFirstd(keys),
        Spacingd(keys, (spacing, spacing, spacing),
                 mode=('bilinear', 'nearest')),
        Orientationd(keys, axcodes='RAS'),
        ScaleIntensityd("image"),
        EnsureTyped(keys),
        AsDiscreted(keys='label', to_onehot=n_classes)
    ])

    seg_tform = Compose([
        LoadImaged('image'),
        EnsureChannelFirstd('image'),
        Spacingd('image', (spacing, spacing, spacing),
                 mode='bilinear'),
        Orientationd('image', axcodes='RAS'),
        ScaleIntensityd("image"),
        EnsureTyped('image'),
    ])

    post_tform = Compose(
        [Activationsd(keys='pred', softmax=True),
         AsDiscreted(keys='pred', argmax=True, to_onehot=n_classes),
         KeepLargestConnectedComponentd(keys='pred',
                                        applied_labels=list(
                                            range(1, n_classes))
                                        )
         # AsDiscreted(keys=('label','pred'), to_onehot=True, n_classes=2),
         ]
    )

    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    model = UNet(spatial_dims=3, in_channels=1, out_channels=n_classes,
                 channels=(16, 32, 64, 128, 256),
                 strides=(2, 2, 2, 2), num_res_units=2, norm=Norm.BATCH,
                 dropout=0).to(device)

    # model = FlexibleUNet(1, 2, 'efficientnet-b0', spatial_dims=3, act='memswish', upsample='deconv').to(device)

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

        data = [{"image": im, "label": seg} for im, seg in zip(images, segs)]

        if platform.system() == 'Windows':
            num_workers = 0
        else:
            num_workers = os.cpu_count()
        # data_set = Dataset(data, cls.train_tform)
        data_set = CacheDataset(data, cls.train_tform,
                                num_workers=None,
                                as_contiguous=True,
                                copy_cache=True,
                                runtime_cache=False)
        # loader = DataLoader(data_set, batch_size=cls.batch_size, shuffle=True,
        #                     num_workers=0, drop_last=True,
        #                     pin_memory=False)

        loader = ThreadDataLoader(data_set, batch_size=cls.batch_size, shuffle=True,
                            num_workers=1, drop_last=True, use_thread_workers=False,
                            pin_memory=False, repeats=3, buffer_size=1)

        return loader

    @classmethod
    def load_val_data(cls, data_path, persistent=True, test=False):
        if not test:
            path = Path(data_path).joinpath('val')
        else:
            path = Path(data_path).joinpath('test')

        random.seed(0)
        images = sorted(str(p.absolute()) for p in path.glob("*US.nii*"))
        segs = sorted(str(p.absolute()) for p in path.glob("*label.nii*"))
        data = [{"image": im, "label": seg} for im, seg in zip(images, segs)]

        data_set = CacheDataset(data, cls.val_tform, num_workers=None,
                                as_contiguous=True,
                                copy_cache=True)
        # if persistent and not test:
        #     persistent_cache = Path("./persistent_cache")
        #     persistent_cache.mkdir(parents=True, exist_ok=True)
        #     data_set = PersistentDataset(data, cls.val_tform, persistent_cache)
        # else:
        #     data_set = Dataset(data, cls.val_tform)
        loader = DataLoader(data_set, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=False)

        return loader

    @classmethod
    def load_seg_data(cls, path):
        path = Path(path)

        random.seed(0)
        images = [str(p.absolute()) for p in path.glob("*.nii*")]
        data = [{"image": im} for im in images]

        # ds = CacheDataset(d, xform)
        data_set = Dataset(data, cls.seg_tform)
        loader = DataLoader(data_set, batch_size=1, shuffle=False,
                            num_workers=0,
                            pin_memory=torch.cuda.is_available())

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
            return None

        return cls.model

    @classmethod
    def output_tform(cls, x):
        xt = list_data_collate(cls.post_tform(x))
        return xt['pred'].detach(), xt['label'].detach()

    @classmethod
    def train(cls, data, use_val=False, load_checkpoint=None):
        #config.print_config()
        # torch.cuda.memory._record_memory_history(max_entries=100000)

        data_path = Path(data)

        loader = cls.load_train_data(data_path, use_val=use_val)
        val_loader = cls.load_val_data(data_path)

        if load_checkpoint and load_checkpoint.endswith('.md'):
            net = cls.load_model(load_checkpoint)
        else:
            net = cls.model

        trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
        print("Number of trainable parameters: {}".format(trainable_params))

        # loss = DiceCELoss(softmax=True, include_background=True,
        #                   lambda_dice=0.5)

        gdf_loss = GeneralizedDiceFocalLoss(softmax=True)

        # opt = Novograd(net.parameters(), 1e-2, weight_decay=1e-2, eps=1e-7)
        opt = AdamW(net.parameters(), lr=1e-3, eps=1e-7, amsgrad=False, fused=True)

        trainer = SupervisedTrainer(
            device=cls.device,
            max_epochs=cls.train_epochs,
            epoch_length=len(loader) * loader.repeats,
            train_data_loader=loader,
            network=net,
            optimizer=opt,
            loss_function=gdf_loss.forward,
            # decollate=False,
            key_train_metric={
                "train_meandice": MeanDice(output_transform=cls.output_tform)},
            amp=True,
            optim_set_to_none=True,
        )
        cls.trainer = trainer

        # Load checkpoint if defined
        if load_checkpoint and not load_checkpoint.endswith('.md'):
            checkpoint = torch.load(load_checkpoint)
            if checkpoint['trainer']:
                for k in checkpoint['trainer']:
                    trainer.state.__dict__[k] = checkpoint['trainer'][k]
                trainer.state.epoch = (trainer.state.iteration //
                                       trainer.state.epoch_length)
            checkpoint_loader = CheckpointLoader(load_checkpoint,
                                                 {'net': net, 'opt': opt})
            checkpoint_loader.attach(trainer)

            logdir = Path(load_checkpoint).parent

        else:
            logdir = Path(data_path).joinpath('runs')
            dirs = sorted(
                [int(x.name) for x in logdir.iterdir() if x.is_dir()])
            if not dirs:
                logdir = logdir.joinpath('0')
            else:
                logdir = logdir.joinpath(str(int(dirs[-1]) + 1))
            logdir.mkdir(parents=True, exist_ok=True)
            shutil.copy(str(Path(__file__)), str(logdir.joinpath('deepmitral.py')))

        # Adaptive learning rate
        lr_scheduler = StepLR(opt, 5, 0.985)
        lr_handler = LrScheduleHandler(lr_scheduler)
        lr_handler.attach(trainer)

        # adding checkpoint handler to save models
        # (network params and optimizer stats) during training
        checkpoint_handler = CheckpointSaver(str(logdir),
                                             {'net': net, 'opt': opt,
                                              'trainer': trainer}, n_saved=10,
                                             save_final=True,
                                             epoch_level=True,
                                             save_interval=100)
        checkpoint_handler.attach(trainer)

        # StatsHandler prints loss at every iteration
        # and print metrics at every epoch
        train_stats_handler = StatsHandler(
            name='trainer',
            output_transform=lambda x: x[0]['loss'])
        train_stats_handler.attach(trainer)

        tb_writer = SummaryWriter(log_dir=str(logdir))

        # TensorBoardStatsHandler plots loss at every iteration
        # and plots metrics at every epoch, same as StatsHandler
        train_tensorboard_stats_handler = TensorBoardStatsHandler(
            summary_writer=tb_writer,
            output_transform=lambda x: x[0]['loss'],
            tag_name='loss')
        train_tensorboard_stats_handler.attach(trainer)

        gc_handler = GarbageCollector(trigger_event="iteration")
        gc_handler.attach(trainer)

        # Set up validation step
        evaluator = SupervisedEvaluator(
            device=cls.device,
            val_data_loader=val_loader,
            network=net,
            amp=True,
            inferer=SlidingWindowInferer(cls.patch_size, sw_batch_size=12),
            key_val_metric={
                "val_meandice": MeanDice(output_transform=cls.output_tform,
                                         include_background=False)},
            additional_metrics={
                'HausdorffDistance': HausdorffDistance(
                    percentile=95,
                    output_transform=cls.output_tform,
                    include_background=False),
                'AvgSurfaceDistance': SurfaceDistance(
                    output_transform=cls.output_tform,
                    include_background=False)},
        )

        val_stats_handler = StatsHandler(
            name='evaluator',
            # no need to plot loss value, so disable per iteration output
            output_transform=lambda x: None,
            # fetch global epoch number from trainer
            global_epoch_transform=lambda x: trainer.state.epoch)
        val_stats_handler.attach(evaluator)

        # add handler to record metrics to TensorBoard at every validation
        # epoch
        val_tensorboard_stats_handler = TensorBoardStatsHandler(
            summary_writer=tb_writer,
            # no need to plot loss value, so disable per iteration output
            output_transform=lambda x: None,
            # fetch global epoch number from trainer
            global_epoch_transform=lambda x: trainer.state.epoch, )
        val_tensorboard_stats_handler.attach(evaluator)

        val_handler = ValidationHandler(
            validator=evaluator,
            interval=10
        )
        val_handler.attach(trainer)

        logging.basicConfig(stream=sys.stdout, level=logging.INFO)

        # @trainer.on(Events.ITERATION_COMPLETED(every=10))
        # def record_mem_snapshot():
        #     # In this sample, we save the snapshot after running 5 iterations.
        #     #   - Save as many snapshots as you'd like.
        #     #   - Snapshots will save last `max_entries` number of memory events
        #     #     (100,000 in this example)
        #     torch.cuda.memory._dump_snapshot(logdir.joinpath('snapshot.pickle'))



        set_track_meta(False)
        trainer.run()

        # Stop recording memory snapshot history.
        # torch.cuda.memory._record_memory_history(enabled=None)

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

        names = []
        dice = []
        hd_list = []
        sd_list = []

        start = timer()
        net.eval()

        frozen_mod = torch.jit.optimize_for_inference(torch.jit.script(net))



        with torch.no_grad():
            for batch in loader:
                label = batch['label'].to(device)
                out = sliding_window_inference(batch['image'].to(device),
                                               (96, 96, 96), 8, frozen_mod)
                out = cls.post_tform(decollate_batch({'pred': out}))

                dice.append(mean_dice(list_data_collate(out)['pred'].cpu(),
                              label.cpu()).item())
                hd_list.append(hd(list_data_collate(out)['pred'].cpu(),
                              label.cpu()).item())
                sd_list.append(sd(list_data_collate(out)['pred'].cpu(),
                              label.cpu()).item())

                meta_dict = decollate_batch(batch["image"].meta)
                names.append(os.path.basename(meta_dict[0]['filename_or_obj']).split('-')[0])

                for o, m in zip(out, meta_dict):
                    saver(pred(o['pred']), m)

        end = timer()

        print("Metric Mean_Dice: {}".format(mean_dice.aggregate().item()))
        print("Metric Mean_HD: {}".format(hd.aggregate().item() * cls.spacing))
        print("Metric Mean_SD: {}".format(sd.aggregate().item() * cls.spacing))
        print("Elapsed Time: {}s".format(end - start))

        df = pd.DataFrame({
            'Case': names,
            'Dice': dice,
            'HD': hd_list,
            'MASD': sd_list
        })

        df.HD *= cls.spacing
        df.MASD *= cls.spacing

        df.to_csv(logdir.joinpath('metrics.csv'))

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
                    out = sliding_window_inference(batch['image'].to(device),
                                                   cls.patch_size, 16, net)
                out = cls.post_tform(decollate_batch({'pred': out}))
                meta_dict = decollate_batch(batch["image_meta_dict"])
                for o, m in zip(out, meta_dict):
                    saver(pred(o['pred']), m)

    @classmethod
    def handle_sigint(cls, *_):
        if cls.trainer:
            msg = "Ctrl-c was pressed. Stopping run at epoch {}.".format(
                cls.trainer.state.epoch)
            cls.trainer.should_terminate = True
            cls.trainer.should_terminate_single_epoch = True
        else:
            msg = "Ctrl-c was pressed. Stopping run."
            print(msg, flush=True)
            sys.exit(1)
        print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description='DeepMV training')

    subparsers = parser.add_subparsers(help='sub-command help', dest='mode')
    subparsers.required = True

    train_parse = subparsers.add_parser('train', help="Train the network")
    train_parse.add_argument('-load', type=str,
                             help='load from a given checkpoint')
    train_parse.add_argument('-data', type=str,
                             help='data folder. should contain "train" and '
                                  '"val" sub-folders', required=True)
    train_parse.add_argument('-use_val', action='store_true',
                             help='Flag to indicate that training set should '
                                  'include validation data.')

    val_parse = subparsers.add_parser('validate', help='Evaluate the network')
    val_parse.add_argument('load', type=str,
                           help='load from a given checkpoint')
    val_parse.add_argument('-data', type=str,
                           help='data folder. should contain "train" and '
                                '"val" sub-folders')
    val_parse.add_argument('-use_test', action='store_true',
                           help='Run on test data')

    seg_parse = subparsers.add_parser('segment', help='Segment images')
    seg_parse.add_argument('load', type=str,
                           help='load from a given checkpoint')
    seg_parse.add_argument('data', type=str, help='data folder')

    return parser.parse_args()


def main():
    if platform.system() == 'Linux':
        import resource
        rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    signal.signal(signal.SIGINT, DeepMitral.handle_sigint)
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
