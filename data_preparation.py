import os
import pandas as pd
import matplotlib.pyplot as plt

# 1. Process annotated femur and mask images
data_list = []
xrays_dir = "data/femur/"

for xray in os.listdir(xrays_dir):
    xray_path = os.path.join(xrays_dir, xray)
    masks_path = xray_path.replace("femur", "masks")
    data_list.append((xray_path, masks_path))

data_df = pd.DataFrame(data_list, columns=['xray', 'masks'])

# Shuffle randomly (frac=1 shuffles all rows, reset index ensures clean ordering)
data_df = data_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Split into 80 train and 20 validation
train_df = data_df.iloc[:80]
val_df = data_df.iloc[80:]

train_df.to_csv("data/CSVs/train_dataset.csv", index=False)
val_df.to_csv("data/CSVs/val_dataset.csv", index=False)

# 2. Process unannotated test images
test_list = []
test_dir = "data/test_femur/"

for test_xray in os.listdir(test_dir):
    test_path = os.path.join(test_dir, test_xray)
    test_list.append((test_path, None))

test_df = pd.DataFrame(test_list, columns=['xray', 'masks'])
test_df.to_csv("data/CSVs/test_dataset.csv", index=False)

# 3. Combine everything into one full dataset.csv (150 rows total)
full_df = pd.concat([data_df, test_df], ignore_index=True)
full_df.to_csv("data/CSVs/dataset.csv", index=False)

# 4. Plot bar charts

files = {
    'train': 'data/CSVs/train_dataset.csv',
    'val': 'data/CSVs/val_dataset.csv',
    'test': 'data/CSVs/test_dataset.csv'
}

counts = {split: len(pd.read_csv(path)) for split, path in files.items()}

plt.figure(figsize=(6, 4))
plt.bar(counts.keys(), counts.values(), color=['#4C72B0', '#55A868', '#C44E52'])
plt.ylabel('Number of Samples')
plt.title('Dataset Split Distribution')

for key, val in counts.items():
    plt.text(key, val + 1, str(val), ha='center', fontweight='bold')

plt.ylim(0, max(counts.values()) + 10)
plt.tight_layout()
plt.show()