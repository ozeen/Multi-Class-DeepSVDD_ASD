import os
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.dataload import ASDDataset
import utils
from src.model import ClassVDD_newmobile_mel_only
from ensemble_tester import EnsembleTester

sep = os.sep


def build_model(args):
    """
    根据 args 构建一个模型实例，用于加载单个权重。
    所有 ensemble 成员结构相同，仅权重不同。
    """
    net = ClassVDD_newmobile_mel_only(
        num_classes=args.num_classes,
        device=args.device,
        z_dim=128,
    )

    net = net.to(args.device)

    # 若你原来在别处加过 DataParallel，可以在这里一起处理
    if args.dp:
        net = nn.DataParallel(net, device_ids=args.device_ids)

    return net


def main(args):
    # set random seed
    utils.setup_seed(args.random_seed)

    # set device
    cuda = args.cuda
    device_ids = args.device_ids
    args.dp = False
    if not cuda or device_ids is None:
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{device_ids[0]}")
        if len(device_ids) > 1:
            args.dp = True

    # ---------- 数据 & 标签映射（只为 meta2label 和 transform） ----------
    train_dirs = args.train_dirs + args.add_dirs
    args.meta2label, args.label2meta = utils.metadata_to_label(train_dirs)

    train_file_list = []
    for train_dir in train_dirs:
        train_file_list.extend(utils.get_filename_list(train_dir))

    # 只用来拿 transform，不再创建 dataloader / 训练
    train_dataset = ASDDataset(args, train_file_list, load_in_memory=False)

    args.num_classes = len(args.meta2label.keys())
    args.logger.info(f"Num classes: {args.num_classes}")

    # # ---------- 从目录中读取多个权重，构建 ensemble 模型列表 ----------
    # if not hasattr(args, "weight_dir") or args.weight_dir is None:
    #     raise ValueError("请在 config 中设置 weight_dir，用于存放多个模型权重文件。")

    #weight_dir = args.weight_dir
    if not os.path.isdir(args.ensemble_weight_dir):
        raise ValueError(f"权重目录不存在: {args.ensemble_weight_dir}")

    # 只取常见的 checkpoint 后缀
    ckpt_files = [
        f for f in os.listdir(args.ensemble_weight_dir)
        if f.endswith(".pth") or f.endswith(".pth.tar")
    ]
    ckpt_files = sorted(ckpt_files)
    if len(ckpt_files) == 0:
        raise ValueError(f"在目录 {args.ensemble_weight_dir} 下没有找到任何 .pth / .pth.tar 权重文件。")

    nets = []
    args.logger.info(f"将在以下权重文件上进行 ensemble：{ckpt_files}")
    for ckpt_name in ckpt_files:
        ckpt_path = os.path.join(args.ensemble_weight_dir, ckpt_name)

        net = build_model(args)
        state = torch.load(ckpt_path, map_location=args.device)

        # 兼容两种保存方式：{'model': state_dict} 或直接 state_dict
        if isinstance(state, dict) and "model" in state:
            net.load_state_dict(state["model"])
        else:
            net.load_state_dict(state)

        nets.append(net)
        args.logger.info(f"Loaded checkpoint: {ckpt_path}")

    # ---------- EnsembleTester：只做测试 ----------
    tester = EnsembleTester(
        args=args,
        nets=nets,
        transform=train_dataset.transform,
    )

    avg_auc, avg_pauc = tester.test(save=True)
    args.logger.info(
        f"Ensemble result -> AUC: {avg_auc * 100:.3f}, pAUC: {avg_pauc * 100:.3f}"
    )


def run():
    # init config parameters
    params = utils.load_yaml(file_path="./config/config_mel_only_test.yaml")
    parser = argparse.ArgumentParser(description=params["description"])
    for key, value in params.items():
        parser.add_argument(f"--{key}", default=value, type=utils.set_type)

    parser.add_argument('--ensemble_weight_dir', default='weights/ensemble/mel_only_hard_weights', type=str)

    args = parser.parse_args()

    # init logger and writer（只是为了保持结构一致，方便看结果）
    time_str = time.strftime("%Y-%m-%d-%H", time.localtime(time.time()))
    args.version = f"{args.version}"
    args.version = (
        f"{time_str}-{args.version}"
        if not args.load_epoch and args.time_version
        else args.version
    )
    log_dir = f"runs/{args.version}"
    writer = SummaryWriter(log_dir=log_dir)
    logger = utils.get_logger(filename=os.path.join(log_dir, "running.log"))



    args.writer, args.logger = writer, logger
    args.logger.info(args.version)


    main(args)


if __name__ == "__main__":
    run()
