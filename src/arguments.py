from argparse import ArgumentParser


def add_basic_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--dataset",
        default="cifar10",
        type=str,
        choices=["cifar10", "svhn", "catsvsdogs",
                 "in9l", "mnist", "waterbirds", "celeba", "imagenet"],
    )
    parser.add_argument(
        "--workers",
        default=8,
        type=int,
        metavar="N",
        help="number of data loading workers (default: 4)",
    )
    parser.add_argument(
        "--train",
        default=False,
        action="store_true",
        help="train the model.",
    )
    parser.add_argument(
        "--resume",
        default=False,
        action="store_true",
        help="resume the training.",
    )
    parser.add_argument(
        "--test_batch", default=128, type=int, metavar="N", help="test batchsize"
    )
    parser.add_argument(
        "--arch",
        metavar="ARCH",
        default="small_cnn",
        choices=["resnet50", "resnet18", "resnet32", "small_cnn"],
    )
    parser.add_argument(
        "--train_bias_conflicting_data_ratio",
        type=float,
        default=0.2,
    )
    parser.add_argument(
        "--test_bias_conflicting_data_ratio",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--base_dir",
        type=str,
        metavar="PATH",
        help="base directory to save data and experiments",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        metavar="PATH",
    )
    parser.add_argument(
        "--saved_mask_dir",
        default=None,
        type=str,
        metavar="PATH",
        help="path to the saved augmented data",
    )
    parser.add_argument(
        "--best_erm_model_checkpoint_path",
        default=None,
        type=str,
        metavar="PATH",
        help="path to the best erm model saved checkpoint",
    )
    parser.add_argument(
        "--last_erm_model_checkpoint_path",
        default=None,
        type=str,
        metavar="PATH",
        help="path to the last erm model saved checkpoint",
    )
    parser.add_argument(
        "--finetuned_model_checkpoint_path",
        default=None,
        type=str,
        metavar="PATH",
        help="path to the last erm model saved checkpoint",
    )
    return parser


def add_optimizer_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--optimizer", default="sgd", choices=["sgd", "adam"])
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=0.1,
        type=float,
        metavar="LR",
        help="initial learning rate",
    )
    parser.add_argument(
        "--schedule",
        type=int,
        nargs="+",
        default=[25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275],
        help="Multiply learning rate by gamma at the scheduled epochs (default: 25,50,75,100,125,150,175,200,225,250,275)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="LR is multiplied by gamma on schedule (default: 0.5)",
    )
    parser.add_argument(
        "--momentum", default=0.9, type=float, metavar="M", help="momentum"
    )
    parser.add_argument(
        "--weight_decay",
        "--wd",
        default=1e-4,
        type=float,
        metavar="W",
        help="weight decay (default: 1e-4)",
    )
    parser.add_argument(
        "--lr_scheduler_name",
        default="multi_step",
        type=str,
        help="learing rate scheduler",
    )
    parser.add_argument(
        "--use_nesterov",
        action="store_true",
        default=False,
        help="use nesterov in sgd",
    )

    return parser


def add_device_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--use_cuda", action="store_true", default=False)
    parser.add_argument("--gpu_ids", type=int, nargs="*", default=[0])
    return parser


def add_train_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--masktune",
                        action="store_true", default=False)
    parser.add_argument(
        "--epochs",
        default=10,
        type=int,
        metavar="N",
        help="number of total epochs to run",
    )
    parser.add_argument(
        "--train_batch", default=128, type=int, metavar="N", help="train batchsize"
    )
    parser.add_argument(
        "--selective_classification", action="store_true", default=False
    )
    parser.add_argument(
        "--use_pretrained_weights", action="store_true", default=False
    )
    return parser


def add_test_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--test_data_types",
        default=["biased"],
        type=str,
        nargs="+",
        choices=["mixed_rand", "mixed_same", "original", "biased", "fg_mask",
                 "mixed_next", "no_fg", "only_bg_b", "only_bg_t", "only_fg", "square"],
    )
    return parser


def add_augmask_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--masking_batch_size",
        type=int,
        default=128,
    )
    return parser


def biased_mnist_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--square_number",
        default=1,
        type=int,
        help="Number of squares to be added to images",
    )
    parser.add_argument(
        "--bias_type",
        type=str,
        default="square",
        choices=["square", "background", "foreground", "none"],
        help="type of bias to be injected into the MNIST data"
    )
    return parser


def selective_classification_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--coverage",
        type=int,
        nargs="+",
        default=[80, 85, 90, 95, 100],
        help="percentage of test data to be covered by the selective classification",
    )
    return parser


def init_train_argparse() -> ArgumentParser:
    parser = ArgumentParser(description="PyTorch MaskTune training")
    parser = add_basic_args(parser)
    parser = add_device_args(parser)
    parser = add_optimizer_args(parser)
    parser = add_train_args(parser)
    parser = add_test_args(parser)
    parser = add_augmask_args(parser)
    parser = biased_mnist_args(parser)
    parser = selective_classification_args(parser)
    return parser
