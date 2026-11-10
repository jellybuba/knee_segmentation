import torch

from model import UNet

from args import get_args
import pandas as pd
import os
from datasets import KneeDataset
from torch.utils.data import DataLoader
from trainer import train_model

# we need arguments
args = get_args()

# loading the csv files
train_set = pd.read_csv(os.path.join(args.csv_dir, 'train_dataset.csv'))
val_set = pd.read_csv(os.path.join(args.csv_dir, 'val_dataset.csv'))

# define our datasets
train_dataset = KneeDataset(train_set)
val_dataset = KneeDataset(val_set)

# create data loaders
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

#for data_batch in train_loader:
#    print()
# Initialize the model
model = UNet(
    input_channels=1,
    output_channels=1,
    pretrained_model=False,
    path_weights='',
    path_pretrained='',
    restore_weights=False
)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(model, train_loader, val_loader, device)
