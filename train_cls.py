import argparse
import datetime
import importlib
import logging
import os
from pathlib import Path
import torch
from dataloader import ModelNetDataLoader
from model_utils import train, validate, inplace_relu
from provider import Logger, save_args, save_model
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
from torch.optim.lr_scheduler import CosineAnnealingLR
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def parse_args():
    '''PARAMETERS'''
    parser = argparse.ArgumentParser('training')
    parser.add_argument('--use_cpu', action='store_true', default=False, help='use cpu mode')
    parser.add_argument('--gpu', type=str, default='0', help='specify gpu device')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size in training')
    parser.add_argument('--model', default='model', help='model name')
    parser.add_argument('--class_num', default=40, type=int, choices=[10, 40], help='training on ModelNet10/40')
    parser.add_argument('--epoch', default=300, type=int, help='number of epoch in training')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate in training')
    parser.add_argument('--num_point', type=int, default=1024, help='Point Number')
    parser.add_argument('--optimizer', type=str, default='SGD', help='optimizer for training')
    parser.add_argument('--log_dir', type=str, default=None, help='experiment root')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='decay rate')
    parser.add_argument('--weight_decay', type=float, default=2e-4, help='decay rate')
    parser.add_argument('--use_normals', action='store_true', default=False, help='use normals')
    parser.add_argument('--use_uniform_sample', action='store_true', default=False, help='use uniform sampiling')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--emb_dims', type=int, default=1024, metavar='N', help='Dimension of embeddings')
    parser.add_argument('--k', type=int, default=20, metavar='N',help='Num of nearest neighbors to use')
    parser.add_argument('--g_k', type=int, default=30, metavar='N',help='Num of global layer to use')
    parser.add_argument('--min_lr', default=0.001, type=float, help='min lr')
    return parser.parse_args()


def main(args):
    def printf(str):
        cout_log.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('classification')
    exp_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.log_dir)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    cout_log = logging.getLogger("Model")
    cout_log.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    cout_log.addHandler(file_handler)
    printf(args)

    '''DATA LOADING'''
    printf('Load dataset ...')
    data_path = '../data/modelnet40_normal_resampled'
    # data_path = '/home/t/文档/git_clone/Point-Transformers/modelnet40_normal_resampled/'

    train_dataset = ModelNetDataLoader(root=data_path, args=args, split='train')
    test_dataset = ModelNetDataLoader(root=data_path, args=args,  split='test')
    trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                                  num_workers=10, drop_last=True)
    testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                                                 num_workers=10)

    '''MODEL LOADING'''
    model = importlib.import_module(args.model)
    classifier = model.get_global(args)
    criterion = model.get_loss
    classifier.apply(inplace_relu)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:

        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate*100, momentum=0.9, weight_decay=args.weight_decay)
    if not args.use_cpu:
        print("$=@" * 50)
        classifier = classifier.to(device)
        # criterion = criterion.to(device)
        if device == torch.device('cuda'):
            classifier = torch.nn.DataParallel(classifier)
            print("$**@" * 50)
            cudnn.benchmark = True

    best_test_acc = 0.  # best test accuracy
    best_train_acc = 0.
    best_test_acc_avg = 0.
    best_train_acc_avg = 0.
    best_test_loss = float("inf")
    best_train_loss = float("inf")
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch
    optimizer_dict = None

    if not os.path.isfile(os.path.join(checkpoints_dir, "last_checkpoint.pth")):
        save_args(args, log_dir)
        logger = Logger(os.path.join(log_dir, 'log.txt'), title="ModelNet" + args.model)
        logger.set_names(["Epoch-Num", 'Learning-Rate',
                          'Train-Loss', 'Train-acc-B', 'Train-acc',
                          'Valid-Loss', 'Valid-acc-B', 'Valid-acc'])

    else:
        printf(f"Resuming last checkpoint from {checkpoints_dir}")
        checkpoint_path = os.path.join(checkpoints_dir, "last_checkpoint.pth")
        checkpoint = torch.load(checkpoint_path)
        classifier.load_state_dict(checkpoint['net'])
        start_epoch = checkpoint['epoch']
        best_test_acc = checkpoint['best_test_acc']
        best_train_acc = checkpoint['best_train_acc']
        best_test_acc_avg = checkpoint['best_test_acc_avg']
        best_train_acc_avg = checkpoint['best_train_acc_avg']
        best_test_loss = checkpoint['best_test_loss']
        best_train_loss = checkpoint['best_train_loss']
        logger = Logger(os.path.join(log_dir, 'log.txt'), title="ModelNet" + args.model, resume=True)
        optimizer_dict = checkpoint['optimizer']

    printf('==> Preparing data..')

    if optimizer_dict is not None:
        optimizer.load_state_dict(optimizer_dict)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.3)
    scheduler = CosineAnnealingLR(optimizer, args.epoch, eta_min=args.min_lr)
    for epoch in range(start_epoch, args.epoch):
        printf('Epoch(%d/%s) Learning Rate %s:' % (epoch + 1, args.epoch, optimizer.param_groups[0]['lr']))
        train_out = train(classifier, trainDataLoader, optimizer, criterion,
                          device)  # {"loss", "acc", "acc_avg", "time"}
        test_out = validate(classifier, testDataLoader, criterion, device)
        scheduler.step()

        if test_out["acc"] > best_test_acc:
            best_test_acc = test_out["acc"]
            is_best = True
        else:
            is_best = False

        best_test_acc = test_out["acc"] if (test_out["acc"] > best_test_acc) else best_test_acc
        best_train_acc = train_out["acc"] if (train_out["acc"] > best_train_acc) else best_train_acc
        best_test_acc_avg = test_out["acc_avg"] if (test_out["acc_avg"] > best_test_acc_avg) else best_test_acc_avg
        best_train_acc_avg = train_out["acc_avg"] if (train_out["acc_avg"] > best_train_acc_avg) else best_train_acc_avg
        best_test_loss = test_out["loss"] if (test_out["loss"] < best_test_loss) else best_test_loss
        best_train_loss = train_out["loss"] if (train_out["loss"] < best_train_loss) else best_train_loss

        save_model(
            classifier, epoch, path=checkpoints_dir, acc=test_out["acc"], is_best=is_best,
            best_test_acc=best_test_acc,  # best test accuracy
            best_train_acc=best_train_acc,
            best_test_acc_avg=best_test_acc_avg,
            best_train_acc_avg=best_train_acc_avg,
            best_test_loss=best_test_loss,
            best_train_loss=best_train_loss,
            optimizer=optimizer.state_dict()
        )
        logger.append([epoch, optimizer.param_groups[0]['lr'],
                       train_out["loss"], train_out["acc_avg"], train_out["acc"],
                       test_out["loss"], test_out["acc_avg"], test_out["acc"]])
        printf(
            f"Training loss:{train_out['loss']} acc_avg:{train_out['acc_avg']}% acc:{train_out['acc']}% time:{train_out['time']}s")
        printf(
            f"Testing loss:{test_out['loss']} acc_avg:{test_out['acc_avg']}% "
            f"acc:{test_out['acc']}% time:{test_out['time']}s [best test acc: {best_test_acc}%] \n\n")
    logger.close()

    printf(f"++++++++" * 2 + "Final results" + "++++++++" * 2)
    printf(f"++  Last Train time: {train_out['time']} | Last Test time: {test_out['time']}  ++")
    printf(f"++  Best Train loss: {best_train_loss} | Best Test loss: {best_test_loss}  ++")
    printf(f"++  Best Train acc_B: {best_train_acc_avg} | Best Test acc_B: {best_test_acc_avg}  ++")
    printf(f"++  Best Train acc: {best_train_acc} | Best Test acc: {best_test_acc}  ++")
    printf(f"++++++++" * 5)


if __name__ == '__main__':
    args = parse_args()
    main(args)