import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from args import get_args
from datasets import KneeDataset, read_array
from model import UNet
from utils import dice_loss_from_logits


def plot_training_history(history_path, out_dir):
    if not os.path.isfile(history_path):
        print(f"No training history found at '{history_path}'. Skipping training curves.")
        return

    df = pd.read_csv(history_path)
    epochs = df['epoch']
    best_loss_idx = df['val_loss'].idxmin()
    best_dice_idx = df['val_dice'].idxmax()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Loss curve
    axes[0].plot(epochs, df['train_loss'], marker='o', markersize=3, label='Train Loss')
    axes[0].plot(epochs, df['val_loss'], marker='o', markersize=3, label='Val Loss')
    axes[0].axvline(df.loc[best_loss_idx, 'epoch'], color='red', linestyle='--', alpha=0.7,
                    label=f"Best Val Loss (epoch {int(df.loc[best_loss_idx, 'epoch'])})")
    axes[0].set_title('Training & Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (BCE + Dice)')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Dice curve
    axes[1].plot(epochs, df['train_dice'], marker='o', markersize=3, label='Train Dice')
    axes[1].plot(epochs, df['val_dice'], marker='o', markersize=3, label='Val Dice')
    axes[1].axvline(df.loc[best_dice_idx, 'epoch'], color='green', linestyle='--', alpha=0.7,
                    label=f"Best Val Dice (epoch {int(df.loc[best_dice_idx, 'epoch'])})")
    axes[1].set_title('Training & Validation Dice Score')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice Score')
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    save_path = os.path.join(out_dir, 'training_curves.png')
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print("=== Training History Summary ===")
    print(f"Total epochs trained : {len(df)}")
    print(f"Best Val Loss        : {df.loc[best_loss_idx, 'val_loss']:.4f} at epoch {int(df.loc[best_loss_idx, 'epoch'])}")
    print(f"Best Val Dice        : {df.loc[best_dice_idx, 'val_dice']:.4f} at epoch {int(df.loc[best_dice_idx, 'epoch'])}")
    print(f"Saved training curves to '{save_path}'\n")


def compute_batch_metrics(logits, targets, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()
    targets = targets.float()

    dims = (1, 2, 3)
    tp = (preds * targets).sum(dims)
    fp = (preds * (1.0 - targets)).sum(dims)
    fn = ((1.0 - preds) * targets).sum(dims)
    tn = ((1.0 - preds) * (1.0 - targets)).sum(dims)

    dice = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
    iou = (tp + eps) / (tp + fp + fn + eps)
    acc = (tp + tn) / (tp + fp + fn + tn + eps)
    return dice.cpu().numpy(), iou.cpu().numpy(), acc.cpu().numpy(), probs.cpu().numpy()


def make_error_overlay(xray, gt, pred):
    rgb = np.stack([xray, xray, xray], axis=-1)
    overlay = rgb.copy()
    tp = (gt > 0.5) & (pred > 0.5)
    fp = (gt <= 0.5) & (pred > 0.5)
    fn = (gt > 0.5) & (pred <= 0.5)

    overlay[tp] = [0.2, 0.85, 0.2]   # Green: true positive
    overlay[fp] = [0.9, 0.2, 0.2]    # Red: false positive
    overlay[fn] = [0.2, 0.4, 0.95]   # Blue: false negative
    return np.clip(0.55 * rgb + 0.45 * overlay, 0, 1)


@torch.no_grad()
def evaluate_validation(model, val_df, device, batch_size, out_dir):
    model.eval()
    val_dataset = KneeDataset(val_df)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    bce = nn.BCEWithLogitsLoss()

    running_loss = 0.0
    all_dice, all_iou, all_acc = [], [], []
    samples = []

    for batch in val_loader:
        images = batch['xray'].to(device)
        masks = batch['mask'].to(device)
        logits = model(images)

        loss = bce(logits, masks) + dice_loss_from_logits(logits, masks)
        running_loss += loss.item() * images.size(0)

        dice, iou, acc, probs = compute_batch_metrics(logits, masks)
        all_dice.extend(dice.tolist())
        all_iou.extend(iou.tolist())
        all_acc.extend(acc.tolist())

        imgs_np = images.cpu().numpy()
        masks_np = masks.cpu().numpy()
        for i in range(images.size(0)):
            samples.append({
                'xray': imgs_np[i, 0],
                'gt': masks_np[i, 0],
                'prob': probs[i, 0],
                'dice': float(dice[i]),
                'iou': float(iou[i]),
            })

    mean_loss = running_loss / len(val_dataset)
    print("=== Validation Evaluation ===")
    print(f"Samples        : {len(val_dataset)}")
    print(f"Val Loss       : {mean_loss:.4f}")
    print(f"Mean Dice      : {np.mean(all_dice):.4f} +/- {np.std(all_dice):.4f}")
    print(f"Mean IoU       : {np.mean(all_iou):.4f} +/- {np.std(all_iou):.4f}")
    print(f"Pixel Accuracy : {np.mean(all_acc):.4f} +/- {np.std(all_acc):.4f}")

    # Plot per-sample Dice distribution
    fig_dist, ax = plt.subplots(figsize=(8, 3.8))
    ax.bar(range(len(all_dice)), sorted(all_dice), color='#4C72B0')
    ax.axhline(np.mean(all_dice), color='red', linestyle='--', label=f"Mean Dice = {np.mean(all_dice):.4f}")
    ax.set_title('Validation Per-Sample Dice Scores (Sorted)')
    ax.set_xlabel('Validation Sample (sorted)')
    ax.set_ylabel('Dice Score')
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis='y', alpha=0.3)
    ax.legend()
    plt.tight_layout()
    dist_path = os.path.join(out_dir, 'val_dice_distribution.png')
    plt.savefig(dist_path, dpi=150)
    plt.close(fig_dist)

    # Visualize representative validation predictions (worst -> best evenly spaced)
    samples_sorted = sorted(samples, key=lambda s: s['dice'])
    indices = np.linspace(0, len(samples_sorted) - 1, min(5, len(samples_sorted)), dtype=int)
    chosen = [samples_sorted[i] for i in indices]

    fig_vis, axes = plt.subplots(len(chosen), 4, figsize=(12, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = np.expand_dims(axes, axis=0)

    col_titles = ['Input X-ray', 'Ground Truth', 'Predicted Mask', 'Overlay (G=TP, R=FP, B=FN)']
    for row, s in enumerate(chosen):
        pred_bin = (s['prob'] > 0.5).astype(np.float32)
        overlay = make_error_overlay(s['xray'], s['gt'], pred_bin)

        axes[row, 0].imshow(s['xray'], cmap='gray')
        axes[row, 1].imshow(s['gt'], cmap='gray')
        axes[row, 2].imshow(pred_bin, cmap='gray')
        axes[row, 3].imshow(overlay)

        axes[row, 0].set_ylabel(f"Dice: {s['dice']:.3f}\nIoU: {s['iou']:.3f}", fontsize=9)
        for col in range(4):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
            if row == 0:
                axes[row, col].set_title(col_titles[col], fontsize=10)

    plt.tight_layout()
    vis_path = os.path.join(out_dir, 'val_predictions.png')
    plt.savefig(vis_path, dpi=150)
    plt.close(fig_vis)

    print(f"Saved validation Dice chart to '{dist_path}'")
    print(f"Saved validation predictions to '{vis_path}'\n")


@torch.no_grad()
def predict_test_samples(model, test_df, device, out_dir, num_vis=6):
    model.eval()
    mask_dir = os.path.join(out_dir, 'test_pred_masks')
    os.makedirs(mask_dir, exist_ok=True)

    vis_items = []
    for idx, row in test_df.iterrows():
        xray_path = row['xray']
        xray = read_array(xray_path)
        tensor = torch.from_numpy(xray).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        pred_mask = (prob > 0.5).astype(np.uint8) * 255

        cv2.imwrite(os.path.join(mask_dir, os.path.basename(xray_path)), pred_mask)
        if len(vis_items) < num_vis:
            vis_items.append((os.path.basename(xray_path), xray[0], (prob > 0.5).astype(np.float32)))

    if vis_items:
        fig, axes = plt.subplots(len(vis_items), 3, figsize=(9, 3 * len(vis_items)))
        if len(vis_items) == 1:
            axes = np.expand_dims(axes, axis=0)

        for row, (name, xray_img, pred_bin) in enumerate(vis_items):
            rgb = np.stack([xray_img, xray_img, xray_img], axis=-1)
            overlay = rgb.copy()
            overlay[pred_bin > 0.5] = [0.2, 0.85, 0.2]
            blended = np.clip(0.6 * rgb + 0.4 * overlay, 0, 1)

            axes[row, 0].imshow(xray_img, cmap='gray')
            axes[row, 1].imshow(pred_bin, cmap='gray')
            axes[row, 2].imshow(blended)
            axes[row, 0].set_ylabel(name, fontsize=9)

            for col, title in enumerate(['Test X-ray', 'Predicted Mask', 'Overlay']):
                axes[row, col].set_xticks([])
                axes[row, col].set_yticks([])
                if row == 0:
                    axes[row, col].set_title(title, fontsize=10)

        plt.tight_layout()
        test_vis_path = os.path.join(out_dir, 'test_predictions.png')
        plt.savefig(test_vis_path, dpi=150)
        plt.close(fig)
        print(f"=== Test Inference ===")
        print(f"Saved {len(test_df)} predicted test masks to '{mask_dir}'")
        print(f"Saved test prediction grid to '{test_vis_path}'")


def main():
    args = get_args()
    os.makedirs(args.out_dir, exist_ok=True)

    history_path = os.path.join(args.out_dir, 'history.csv')
    plot_training_history(history_path, args.out_dir)

    weights_path = os.path.join(args.out_dir, 'model.pth')

    if not os.path.isfile(weights_path):
        print(f"No trained weights found in '{args.out_dir}'. Run 'python main.py' first.")
        return

    print(f"Loading model weights from '{weights_path}'...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = UNet(
        input_channels=1,
        output_channels=1,
        restore_weights=True,
        path_weights=weights_path,
    ).to(device)

    val_csv = os.path.join(args.csv_dir, 'val_dataset.csv')
    if os.path.isfile(val_csv):
        val_df = pd.read_csv(val_csv)
        evaluate_validation(model, val_df, device, args.batch_size, args.out_dir)

    test_csv = os.path.join(args.csv_dir, 'test_dataset.csv')
    if os.path.isfile(test_csv):
        test_df = pd.read_csv(test_csv)
        predict_test_samples(model, test_df, device, args.out_dir)


if __name__ == '__main__':
    main()
