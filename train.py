import argparse
import glob
import json
import multiprocessing
import os
import random
import re
from importlib import import_module
from pathlib import Path
from copy import copy

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from sklearn.model_selection import StratifiedKFold

from dataset import MaskBaseDataset
from model import BaseModel
from loss import create_criterion

from sklearn.metrics import f1_score, accuracy_score, classification_report
from datetime import datetime
from pytz import timezone

import wandb

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def grid_image(np_images, gts, preds, n=16, shuffle=False, task = 'total'):
    batch_size = np_images.shape[0]
    assert n <= batch_size

    choices = random.choices(range(batch_size), k=n) if shuffle else list(range(n))
    figure = plt.figure(figsize=(12, 18 + 2))
    plt.subplots_adjust(top=0.8)
    n_grid = int(np.ceil(n ** 0.5))
    
    
    if task == 'total':
        tasks = ["mask", "gender", "age"]
        for idx, choice in enumerate(choices):
            gt = gts[choice].item()
            pred = preds[choice].item()
            image = np_images[choice]
            
            gt_decoded_labels = MaskBaseDataset.decode_multi_class(gt)
            pred_decoded_labels = MaskBaseDataset.decode_multi_class(pred)
            
            title = "\n".join([
            
                f"{task} - gt: {gt_label}, pred: {pred_label}"
                for gt_label, pred_label, task
                in zip(gt_decoded_labels, pred_decoded_labels, tasks)
            ])
            plt.subplot(n_grid, n_grid, idx + 1, title=title)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap=plt.cm.binary)
    else:
        for idx, choice in enumerate(choices):
            gt = gts[choice].item()
            pred = preds[choice].item()
            image = np_images[choice]
            tasks = task
            title = '\n{} - gt: {}, pred: {}'.format(task, int(gt), int(pred))

            plt.subplot(n_grid, n_grid, idx + 1, title=title)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(image, cmap=plt.cm.binary)

    return figure


def increment_path(dir, name, task = 'total'):
    path = os.path.join(dir, f'{task[0]}__{name}')
    path = Path(path)

    now = datetime.now(timezone('Asia/Seoul')).strftime('__%y%m%d_%H%M%S')
    wandbname = f'{task[0]}__{name}{now}'
    return f"{path}{now}", wandbname

# https://sseunghyuns.github.io/classification/2021/05/25/invasive-pytorch/#
def rand_bbox(size, lam): # size : [B, C, W, H]
    W = size[2] # 이미지의 width
    H = size[3] # 이미지의 height
    cut_rat = np.sqrt(1. - lam)  # 패치 크기의 비율 정하기
    cut_w = np.int(W * cut_rat)  # 패치의 너비
    cut_h = np.int(H * cut_rat)  # 패치의 높이

    # uniform
    # 기존 이미지의 크기에서 랜덤하게 값을 가져옵니다.(중간 좌표 추출)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    # 패치 부분에 대한 좌표값을 추출합니다.
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2

def getDataloader(dataset, train_idx, valid_idx, batch_size, num_workers):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_set = torch.utils.data.Subset(dataset,
                                        indices=train_idx)
    val_set   = torch.utils.data.Subset(dataset,
                                        indices=valid_idx)
    
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=(dataset.train_weight == None),
        pin_memory=use_cuda,
        drop_last=True,
        sampler=dataset.train_weight,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set,
        batch_size=args.valid_batch_size,
        num_workers=multiprocessing.cpu_count() // 2,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=True,
    )
    
    return train_loader, val_loader

def train(data_dir, model_dir, args):
    
    seed_everything(args.seed)

    save_dir, wandbname = increment_path(model_dir, args.model, args.task)

    # -- settings
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    # -- dataset
    dataset_module = getattr(import_module("dataset"), args.dataset)  # default: MaskBaseDataset
    dataset = dataset_module(
        data_dir=data_dir,
        val_ratio=args.val_ratio,
    )
    num_classes = dataset.num_classes  # 18

    # 5-fold Stratified KFold 5개의 fold를 형성하고 5번 Cross Validation을 진행합니다.
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits)

    counter = 0
    patience = 10
    accumulation_steps = 2
    best_val_acc = 0
    best_val_loss = np.inf

    labels = [dataset.encode_multi_class(mask, gender, age) for mask, gender, age in zip(dataset.mask_labels, dataset.gender_labels, dataset.age_labels)]

    for i, (train_idx, valid_idx) in enumerate(skf.split(dataset.image_paths, labels)):
        
        ## wandb 설정
        wandb.init(
                project=f'{args.wandbproject}',
                entity=f'{args.wandbentity}',
                name = wandbname
            )
        wandb.config.update(args)

        transform_module = getattr(import_module("dataset"), 'BaseAugmentation')
        transform = transform_module(
            resize=args.resize,
            mean=dataset.mean,
            std=dataset.std,
        )
        dataset.set_transform(transform)
        _, val_loader = getDataloader(dataset, train_idx, valid_idx, args.batch_size, multiprocessing.cpu_count() // 2)

        dataset = copy(dataset)
        transform_module = getattr(import_module("dataset"), args.augmentation)  # default: BaseAugmentation
        transform = transform_module(
            resize=args.resize,
            mean=dataset.mean,
            std=dataset.std,
        )
        dataset.set_transform(transform)
        train_loader, _ = getDataloader(dataset, train_idx, valid_idx, args.batch_size, multiprocessing.cpu_count() // 2)
        
        
        sampler = dataset.train_weight

        # -- model
        model = BaseModel(
                num_classes=num_classes,
                model = args.model
                ).to(device)
        model = torch.nn.DataParallel(model)

        # -- loss & metric
        criterion = create_criterion(args.criterion)  # default: cross_entropy
        opt_module = getattr(import_module("torch.optim"), args.optimizer)
        optimizer = opt_module(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr,
            weight_decay=1e-4
        )
        
        scheduler = StepLR(optimizer, args.lr_decay_step, gamma=0.5)

        # -- logging
        logger = SummaryWriter(log_dir=save_dir)
        with open(os.path.join(save_dir, 'config.json'), 'w', encoding='utf-8') as f:
            json.dump(vars(args), f, ensure_ascii=False, indent=4)
        
        best_val_acc = 0
        best_val_loss = np.inf
        best_val_f1 = 0

        for epoch in range(args.epochs):
            # train loop
            model.train()
            loss_value = 0
            matches = 0
            for idx, train_batch in enumerate(train_loader):
                inputs, labels = train_batch
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                
                # cutmix
                if np.random.random()>0.3:
                    lam = np.random.beta(0.5, 1.0)
                    rand_index = torch.randperm(inputs.size()[0]).to(device)
                    target_a = labels
                    target_b = labels[rand_index]
                    bbx1, bby1, bbx2, bby2 = rand_bbox(inputs.size(), lam)
                    inputs[:, :, bbx1:bbx2, bby1:bby2] = inputs[rand_index, :, bbx1:bbx2, bby1:bby2]
                    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (inputs.size()[-1] * inputs.size()[-2]))
                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, target_a) * lam + criterion(outs, target_b) * (1. - lam)
                else:
                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)
                    loss = criterion(outs, labels)

                loss.backward()
                optimizer.step()

                loss_value += loss.item()
                matches += (preds == labels).sum().item()
                if (idx + 1) % args.log_interval == 0:
                    train_loss = loss_value / args.log_interval
                    train_acc = matches / args.batch_size / args.log_interval
                    current_lr = get_lr(optimizer)
                    print(
                        f"Epoch[{epoch}/{args.epochs}]({idx + 1}/{len(train_loader)}) || "
                        f"training loss {train_loss:4.4} || training accuracy {train_acc:4.2%} || lr {current_lr}"
                    )
                    logger.add_scalar("Train/loss", train_loss, epoch * len(train_loader) + idx)
                    logger.add_scalar("Train/accuracy", train_acc, epoch * len(train_loader) + idx)

                    ### WandB ###
                    wandb.log({
                    "Train Accuracy": train_acc,
                    "Train Loss": train_loss})

                    loss_value = 0
                    matches = 0

            scheduler.step()

            # val loop
            with torch.no_grad():
                print("Calculating validation results...")
                model.eval()
                val_loss_items = []
                val_acc_items = []
                figure = None

                targets = []
                predictions = []

                for val_batch in val_loader:
                    inputs, labels = val_batch
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    outs = model(inputs)
                    preds = torch.argmax(outs, dim=-1)

                    targets += labels.tolist()
                    predictions += preds.tolist()

                    loss_item = criterion(outs, labels).item()
                    acc_item = (labels == preds).sum().item()
                    val_loss_items.append(loss_item)
                    val_acc_items.append(acc_item)

                    if figure is None:
                        inputs_np = torch.clone(inputs).detach().cpu().permute(0, 2, 3, 1).numpy()
                        inputs_np = dataset_module.denormalize_image(inputs_np, dataset.mean, dataset.std)
                        figure = grid_image(
                            inputs_np, labels, preds, n=16, shuffle=args.dataset != "MaskSplitByProfileDataset", task = args.task
                        )

                val_loss = np.sum(val_loss_items) / len(val_loader)
                val_acc = accuracy_score(targets, predictions)
                val_f1 = f1_score(targets, predictions, average='macro')
                best_val_loss = min(best_val_loss, val_loss)
                best_val_acc = max(best_val_acc, val_acc)

                print(classification_report(targets, predictions))

                if val_f1 > best_val_f1:
                    print(f"New best model for val macro-f1 : {val_f1:4.2%}! saving the best model..")
                    torch.save(model.module.state_dict(), f"{save_dir}/{i}_best.ckpt")
                    best_val_f1 = val_f1

                print(
                    f"[Val] acc : {val_acc:4.2%}, loss: {val_loss:4.2}, macro-f1: {val_f1:4.2} || "
                    f"best acc : {best_val_acc:4.2%}, best loss: {best_val_loss:4.2}, best macro-f1: {best_val_f1:4.2}"
                )
                logger.add_scalar("Val/loss", val_loss, epoch)
                logger.add_scalar("Val/accuracy", val_acc, epoch)
                logger.add_figure("results", figure, epoch)
                print()

                wandb.log({
                "Val Accuracy": val_acc,
                "Val Loss": val_loss,
                "F1 score": val_f1,
                "Pred Figure": [wandb.Image(figure, caption="Val Pred")]
                })    
                print()

        wandb.log({
            "Best Val Acc" : best_val_acc,
            "Best Val loss" : best_val_loss,
            "Best F1 score" : best_val_f1
            })
        wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--seed', type=int, default=42, help='random seed (default: 777)')
    parser.add_argument('--epochs', type=int, default=1, help='number of epochs to train (default: 1)')
    parser.add_argument('--dataset', type=str, default='TotalTaskDataset', help='dataset augmentation type (default: TotalDataset)')
    parser.add_argument('--augmentation', type=str, default='BaseAugmentation', help='data augmentation type (default: BaseAugmentation)')
    parser.add_argument("--resize", nargs="+", type=list, default=[96, 128], help='resize size for image when training')
    parser.add_argument('--batch_size', type=int, default=64, help='input batch size for training (default: 64)')
    parser.add_argument('--valid_batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--model', type=str, default='resnet50', help='model save at {SM_MODEL_DIR}/{name}')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer type (default: Adam)')
    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate (default: 1e-3)')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='ratio for validaton (default: 0.2)')
    parser.add_argument('--criterion', type=str, default='cross_entropy', help='criterion type (default: cross_entropy)')
    parser.add_argument('--lr_decay_step', type=int, default=20, help='learning rate scheduler deacy step (default: 20)')
    parser.add_argument('--log_interval', type=int, default=20, help='how many batches to wait before logging training status')
    parser.add_argument("--task", type=str, default='total',help='Tensorboard options: total, age, mask, gender (default: total)')
    parser.add_argument("--sampler", type=str, default='no',help='DataLoader sampler type (default: no)')
    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '/opt/ml/input/data/train/images'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', './model'))

    # wandb options
    parser.add_argument("--wandbproject", type=str,
                        default='scv_mask_competition', help='wandb project')
    parser.add_argument("--wandbentity", type=str,
                        default='scv_mask_competition', help='wandb entity')
    
    args = parser.parse_args()
    print(args)

    data_dir = args.data_dir
    model_dir = args.model_dir

    train(data_dir, model_dir, args)