import argparse
import multiprocessing
import os
import numpy as np
from importlib import import_module

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import TestDataset
from model import BaseModel

def load_model(saved_model, num_classes, device, idx):
    model = BaseModel(
        num_classes=num_classes,
        model = saved_model.split('__')[1]
    )

    model_path = os.path.join(saved_model, f'{idx}_best.ckpt')
    model.load_state_dict(torch.load(model_path, map_location=device))

    return model


@torch.no_grad()
def inference(data_dir, model_dir, output_dir, args):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    task = model_dir.split('__')[0][-1]
    if task == 't':
        num_classes = 18
    elif task == 'g':
        num_classes = 2
    else:
        num_classes = 3
    
    info_path = os.path.join(data_dir, 'info.csv')
    submit_info = pd.read_csv(info_path)
    for i in range(5):
        
        model = load_model(model_dir, num_classes, device, i).to(device)
        model.eval()

        img_root = os.path.join(data_dir, 'images')

        img_paths = [os.path.join(img_root, img_id) for img_id in submit_info.ImageID]
        dataset = TestDataset(img_paths, args.resize)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=multiprocessing.cpu_count() // 2,
            shuffle=False,
            pin_memory=use_cuda,
            drop_last=False,
        )

        print(f"Calculating inference results.. fold {i}")
        preds = []
        softvoting = None
        with torch.no_grad():
            for idx, images in enumerate(loader):
                images = images.to(device)
                pred = model(images) / 2
                pred += model(torch.flip(images, dims=(-1,))) / 2
                preds.extend(pred.cpu().numpy())
    
            
        softvoting = softvoting+np.array(preds) if softvoting is not None else np.array(preds)    
        

    submit_info['ans'] = np.argmax(softvoting, axis=1)
    submit_info = submit_info[['ImageID','ans']]
        
    save_path = os.path.join(output_dir, 'final.csv'.format(model_dir.split('/')[-1]))
        
    submit_info.to_csv(save_path, index=False)
    print(f"Inference Done! Inference result saved at {save_path}")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument('--batch_size', type=int, default=1000, help='input batch size for validing (default: 1000)')
    parser.add_argument('--resize', type=tuple, default=(96, 128), help='resize size for image when you trained (default: (96, 128))')

    # Container environment
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_EVAL', '/opt/ml/input/data/eval'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_CHANNEL_MODEL', 'untitled'))
    parser.add_argument('--output_dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', './output'))

    args = parser.parse_args()

    data_dir = args.data_dir
    model_dir = args.model_dir
    output_dir = args.output_dir

    os.makedirs(output_dir, exist_ok=True)

    inference(data_dir, model_dir, output_dir, args)