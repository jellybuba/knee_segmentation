import os
import pandas as pd
from torch import optim, nn

from args import get_args
import torch
from utils import *

def train_model(model, train_loader, val_loader, device):
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    model = model.to(device)

    bce = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'train_dice': [], 'val_dice': []}

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        running_dice = 0.0

        for data_batch in train_loader:
            images = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)

            optimizer.zero_grad()
            outputs = model(images)

            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_dice += dice_score_from_logits(outputs, masks)

        train_loss = running_loss / len(train_loader)
        train_dice = running_dice / len(train_loader)

        val_loss, val_dice = validate_model(model, val_loader, bce, device)
        print(f"Epoch [{epoch + 1}/{args.epochs}] - Train Loss: {train_loss:.4f}, Train Dice: {train_dice:.4f} | Val Loss: {val_loss:.4f}, Val Dice: {val_dice:.4f}")

        history['epoch'].append(epoch + 1)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_dice'].append(train_dice)
        history['val_dice'].append(val_dice)

    torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pth'))
    pd.DataFrame(history).to_csv(os.path.join(args.out_dir, 'history.csv'), index=False)
    return history

def validate_model(model, val_loader, bce, device):
    model.eval()
    running_loss = 0.0
    running_dice = 0.0
    with torch.no_grad():
        for data_batch in val_loader:
            images = data_batch['xray'].to(device)
            masks = data_batch['mask'].to(device)
            outputs = model(images)
            loss_bce = bce(outputs, masks)
            loss_dice = dice_loss_from_logits(outputs, masks)
            loss = loss_bce + loss_dice

            running_loss += loss.item()
            running_dice += dice_score_from_logits(outputs, masks)

    val_loss = running_loss / len(val_loader)
    val_dice = running_dice / len(val_loader)

    return val_loss, val_dice
