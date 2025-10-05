from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import lightning as L
from lightning.pytorch import callbacks as lightning_callbacks
from lightning.pytorch.loggers import CSVLogger, WandbLogger

from jurassic_open_world.data.datamodule import (
    TextBasedDetectionDataModuleConfig,
)
from jurassic_open_world.modeling.model.grounding_dino import (
    GroundingDINOConfig,
)
from jurassic_open_world.training.training import GroundingDINOTrainingConfig
from jurassic_open_world.utils import callbacks as custom_callbacks


def parse_args():
    parser = ArgumentParser(
        "Grounding DINO pytorch lightning training script."
    )
    parser.add_argument(
        "--data-config",
        type=str,
        default="configs/data/coco.yaml",
        help="Path to data config in .yaml format. Refer to jurassic_open_world.config.data.DataConfig for format.",
    )
    parser.add_argument(
        "--model-config",
        type=str,
        default="configs/model/idea_grounding_dino_tiny.yaml",
        help="Path to model config in .yaml format. Refer to jurassic_open_world.config.model.GroundingDINOConfig for format.",
    )
    parser.add_argument(
        "--training-config",
        type=str,
        default="configs/training/finetune_1x.yaml",
        help="Path to training config in .yaml format. Refer to jurassic_open_world.config.training.TrainingConfig for format.",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        default=None,
        help="Path to model state dict from which to initialize.",
    )
    parser.add_argument(
        "--save-path",
        type=Path,
        default=Path("outputs"),
        help="Path to dir to save outputs (logs and checkpoints) to.",
    )
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Whether to train in Automatic Mixed Precision mode.",
    )
    parser.add_argument(
        "--num-nodes",
        type=int,
        default=1,
        help="Number of nodes to use for training.",
    )
    parser.add_argument(
        "--gpus-per-node",
        type=int,
        default=1,
        help="Number of gpus to use for training.",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Set to use Weights&Biases logger. Uses CSVLogger otherwise.",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default="grounding-dino-pl",
        help="Project name for Weights&Biases logger.",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="grounding-dino-pl",
        help="Run name for Weights&Biases logger.",
    )
    parser.add_argument(
        "--img-log-freq",
        type=int,
        default=5_000,
        help="Frequency (in steps) for image logging.",
    )
    parser.add_argument(
        "--continue-checkpoint",
        type=Path,
        default=None,
        help="Path to checkpoint from which to continue training.",
    )
    parser.add_argument(
        "--val-check-interval",
        type=float,
        default=1.0,
        help="Validation check interval for L.Trainer.",
    )
    parser.add_argument("--seed", type=int, default=1337, help="Rng seed.")

    return parser.parse_args()


def main(args):
    # check if we are running in distributed
    is_distributed = args.num_nodes != 1 or args.gpus_per_node != 1

    # load config files
    data_config = TextBasedDetectionDataModuleConfig.from_yaml(
        args.data_config
    )
    model_config = GroundingDINOConfig.from_yaml(args.model_config)
    training_config = GroundingDINOTrainingConfig.from_yaml(
        args.training_config
    )

    # set up logger
    log_save_path = args.save_path / "logs"
    if args.use_wandb:
        logger = WandbLogger(
            name=args.run_name,
            save_dir=log_save_path,
            project=args.project_name,
        )
    else:
        logger = CSVLogger(
            name="csv_logs",
            save_dir=log_save_path,
        )

    # set up callbacks
    checkpoint_save_path = args.save_path / "checkpoints"
    callbacks = [
        custom_callbacks.LossLogger(
            train_kwargs=dict(on_step=True, on_epoch=True, sync_dist=True),
            val_kwargs=dict(on_step=False, on_epoch=True, sync_dist=True),
        ),
        custom_callbacks.VisualizationLogger(args.img_log_freq),
        custom_callbacks.CountingMetricsLogger(
            "mae", 0.3, autotune_threshold=False
        ),
        custom_callbacks.CountingMetricsLogger(
            "rmse", 0.3, autotune_threshold=False
        ),
        custom_callbacks.CountingMetricsLogger(
            "mape", 0.3, autotune_threshold=False
        ),
        (
            custom_callbacks.DistributedMeanAveragePrecisionLogger()
            if is_distributed
            else custom_callbacks.MeanAveragePrecisionLogger()
        ),
        lightning_callbacks.LearningRateMonitor(logging_interval="step"),
        lightning_callbacks.ModelCheckpoint(
            monitor="val/map",
            mode="max",
            save_last=True,
            verbose=True,
            dirpath=checkpoint_save_path,
            filename=f"{args.run_name}-"
            + "{epoch}-{step}"
            + f"-time={datetime.now()}",
        ),
    ]

    # seed if necessary
    L.seed_everything(args.seed)

    # init datamodule
    datamodule = data_config.build()

    # init model, load weights if necessary
    model = model_config.build()
    if args.init_checkpoint is not None:
        model = model.from_pretrained(args.init_checkpoint)
    training_module = training_config.build(model)

    # init trainer
    trainer = L.Trainer(
        strategy="ddp_find_unused_parameters_true"
        if is_distributed
        else "auto",
        logger=logger,
        max_epochs=training_config.num_epochs,
        gradient_clip_val=training_config.grad_clip_val,
        callbacks=callbacks,
        num_nodes=args.num_nodes,
        devices=args.gpus_per_node,
        precision="16-mixed" if args.use_amp else "32-true",
        val_check_interval=args.val_check_interval,
    )

    # run the training
    trainer.fit(
        model=training_module,
        datamodule=datamodule,
        ckpt_path=args.continue_checkpoint,
    )


if __name__ == "__main__":
    main(parse_args())
