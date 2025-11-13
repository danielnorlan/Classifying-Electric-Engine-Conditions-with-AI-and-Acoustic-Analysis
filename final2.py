"""
Filename: final.py
Author: Elias Kamalsen
Date: 30.10.2025
Description: Electric engine sound classification system using CNN and log-mel spectrograms
Course: VE3451 1 AI og maskinlæring 25H Kongsberg
Institution: University of South-Eastern Norway
"""

# ================================================================================================================================
# CODE DESCRIPTION: ENGINE SOUND CLASSIFICATION SYSTEM
# ================================================================================================================================
# This system implements an audio classification pipeline for electric engine condition monitoring, transforming raw WAV
# files into log-mel spectrograms and processing them through a convolutional neural network. The model distinguishes between
# three engine states. The implementation features deterministic dataset splitting, comprehensive multi run evaluation with
# statistical reporting, and detailed subcategory analysis to identify specific failure patterns. Through systematic
# experimentation with controlled randomness, the system provides performance metrics and insights into model behavior across
# different engine operating conditions.

# Data Flow:
# Audio Loading → Load WAV files and use all files in train_cut and test_cut
# Feature Extraction → Split into 1 second audio segments and convert each segments to log-mel spectrograms
# Model Training → Train CNN with batch normalization and adaptive pooling
# Evaluation → Track and display accuracy results for each individual training run
# Reporting → Calculate final averages and standard deviations across all runs

# ================================================================================================================================
# IMPORTS
# ================================================================================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import numpy as np
import random
import os
from collections import defaultdict

from configs import Config
from pathlib import Path

import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass




cfg= Config()
classes = cfg.get_folders()

# ================================================================================================================================
# CONFIGURATION / CONSTANTS
# ================================================================================================================================
RUNS = 1
BATCH_SIZE = 16
SEGMENT_SEC = 1
N_EPOCHS = 10
LR = 0.001
WEIGHT_DECAY = 1e-4

# ================================================================================================================================
# UTILITY FUNCTIONS
# ================================================================================================================================
# --------------------------------------------------------------------------------------------------------------------------------
# Initialize random seeds for Python random, NumPy, and PyTorch
# --------------------------------------------------------------------------------------------------------------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --------------------------------------------------------------------------------------------------------------------------------
# Convert raw audio waveform to log-mel spectrogram
# --------------------------------------------------------------------------------------------------------------------------------
def wav_to_log_mel(y, sr=32000, n_fft=1024, hop_length=320, n_mels=64, eps=1e-10):
    spec = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=hop_length))**2
    mel_basis = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=n_mels)
    mel_spec = np.dot(mel_basis, spec)
    log_mel_spec = np.log(mel_spec + eps)
    x = torch.tensor(log_mel_spec, dtype=torch.float).unsqueeze(0)
    return x

# --------------------------------------------------------------------------------------------------------------------------------
# Extract and pair audio files with labels from train and test directories
# --------------------------------------------------------------------------------------------------------------------------------
def extract_files(train_dirs, test_dirs):
    # Collect all train files
    train_files = []
    for label, cls_dir in enumerate(train_dirs):
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".wav")])
        for f in files:
            train_files.append((os.path.join(cls_dir, f), label))
    print(f"> Total train files: {len(train_files)}")

    # Collect all test files
    test_files = []
    for label, cls_dir in enumerate(test_dirs):
        files = sorted([f for f in os.listdir(cls_dir) if f.endswith(".wav")])
        for f in files:
            test_files.append((os.path.join(cls_dir, f), label))
    print(f"> Total test files:  {len(test_files)}")

    return train_files, test_files

# --------------------------------------------------------------------------------------------------------------------------------
# Extract subcategory name from filename
# --------------------------------------------------------------------------------------------------------------------------------
def get_subcategory(filename):
    base = os.path.basename(filename)
    return "_".join(base.split("_")[:-1])

# ================================================================================================================================
# MAIN CLASSES
# ================================================================================================================================
# --------------------------------------------------------------------------------------------------------------------------------
# Dataset loader that segments audio files and converts to log-mel spectrograms
# --------------------------------------------------------------------------------------------------------------------------------
class EngineDataset(Dataset):
    def __init__(self, file_list, sr=32000, segment_sec=None, transform=wav_to_log_mel, mode="unkown"):
        self.sr = sr
        self.segment_sec = segment_sec
        self.transform = transform or (lambda y, sr: torch.tensor(y, dtype=torch.float).unsqueeze(0))
        self.samples = []

        class_segments = defaultdict(int)

        for file_path, label in file_list:
            y, _ = librosa.load(file_path, sr=sr, mono=True)

            if segment_sec is not None:
                n_samples_per_segment = int(sr * segment_sec)
                n_segments = max(1, len(y) // n_samples_per_segment)
                
                for i in range(n_segments):
                    start = i * n_samples_per_segment
                    end = start + n_samples_per_segment
                    segment = y[start:end]
                    if len(segment) < n_samples_per_segment:
                        segment = np.pad(segment, (0, n_samples_per_segment - len(segment)))
                    self.samples.append((segment, label, file_path))
                    class_segments[label] += 1
            else:
                # No segmentation, store full audio
                self.samples.append((y, label, file_path))
                class_segments[label] += 1

        # Print summary for verification
        print(f"> {mode.upper()} dataset segments per class")
        for label in sorted(class_segments):
            print(f"  Class {label}: {class_segments[label]} segments")
        print(f"  Total segments: {len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        segment, label, file_path = self.samples[idx]
        x = self.transform(segment, sr=self.sr)
        y_tensor = torch.tensor(label, dtype=torch.long)
        return x, y_tensor, file_path

# --------------------------------------------------------------------------------------------------------------------------------
# CNN model for engine sound classification from spectrograms
# --------------------------------------------------------------------------------------------------------------------------------
class EngineCNN(nn.Module):
    def __init__(self, n_classes):
        super(EngineCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(3,3), padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d((2,2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3,3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d((2,2))

        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3,3), padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AdaptiveMaxPool2d((1,1))

        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool3(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ================================================================================================================================
# MAIN EXECUTION
# ================================================================================================================================
# --------------------------------------------------------------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------------------------------------------------------------
# classes = [
#     "engine1_good",
#     "engine2_broken",
#     "engine3_heavyload"
#     ]

# train_dirs = [
#     "C:/Users/Speed/Downloads/ai_project/IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine1_good/",
#     "C:/Users/Speed/Downloads/ai_project/IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine2_broken/",
#     "C:/Users/Speed/Downloads/ai_project/IDMT-ISA-ELECTRIC-ENGINE/train_cut/engine3_heavyload/"
#     ]

# test_dirs = [
#     "C:/Users/Speed/Downloads/ai_project/IDMT-ISA-ELECTRIC-ENGINE/test_cut/engine1_good/",
#     "C:/Users/Speed/Downloads/ai_project/IDMT-ISA-ELECTRIC-ENGINE/test_cut/engine2_broken/",
#     "C:/Users/Speed/Downloads/ai_project/IDMT-ISA-ELECTRIC-ENGINE/test_cut/engine3_heavyload/"
#     ]
train_dirs = [cfg.get_train_cut() / cls for cls in classes]
test_dirs  = [cfg.get_test_cut()  / cls for cls in classes]

print("\n> Directory configuration:")
for cls, tr, te in zip(classes, train_dirs, test_dirs):
    print(f"  {cls}:")
    print(f"    train_cut → {tr}")
    print(f"    test_cut  → {te}")

# --------------------------------------------------------------------------------------------------------------------------------
# Store results across multiple runs for statistical analysis
# --------------------------------------------------------------------------------------------------------------------------------
saved = []                          # Overall test accuracy per run
good = []                           # Class 0 (good) accuracy per run  
broken = []                         # Class 1 (broken) accuracy per run
heavyload = []                      # Class 2 (heavyload) accuracy per run
subcat_summary = defaultdict(list)  # Per-subcategory accuracy per run

# --------------------------------------------------------------------------------------------------------------------------------
# Prepare and process data
# --------------------------------------------------------------------------------------------------------------------------------
train_files, test_files = extract_files(train_dirs, test_dirs)

dataset_train = EngineDataset(train_files, segment_sec=SEGMENT_SEC, mode='train')
dataset_test = EngineDataset(test_files, segment_sec=SEGMENT_SEC, mode='test')

dataloader_train = DataLoader(dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

seed_gen = random.Random()
for each_run in range(RUNS):
    run_seed = seed_gen.randint(0, 999999)
    set_seed(run_seed)
    
    print(f"\n{'='*128}")
    print(f"RUN {each_run + 1}/{RUNS} || SEED {run_seed}")
    print(f"{'='*128}\n")

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = EngineCNN(len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

# --------------------------------------------------------------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------------------------------------------------------------
    for epoch in range(N_EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for x_batch, y_batch, file_path in dataloader_train:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            correct_train += (predicted == y_batch).sum().item()
            total_train += y_batch.size(0)
        
        avg_loss = running_loss / len(dataloader_train)
        train_acc = 100 * correct_train / total_train

        print(f"> Epoch [{epoch+1:2d}/{N_EPOCHS}] | Loss: {avg_loss:.4f} | Train accuracy: {train_acc:.2f}%")

# --------------------------------------------------------------------------------------------------------------------------------
# Testing
# --------------------------------------------------------------------------------------------------------------------------------    
    model.eval()
    test_correct = 0
    test_total = 0

    test_class_correct = [0] * len(classes)
    test_class_total = [0] * len(classes)
    subcat_total_per_class = {cls: defaultdict(int) for cls in classes}
    subcat_correct_per_class = {cls: defaultdict(int) for cls in classes}
    subcat_correct = {label: defaultdict(int) for label in range(len(classes))}
    subcat_total = {label: defaultdict(int) for label in range(len(classes))}

    with torch.no_grad():
        for x_test, y_test, file_paths in dataloader_test:
            x_test, y_test = x_test.to(device), y_test.to(device)
            outputs = model(x_test)
            _, predicted = torch.max(outputs, 1)

            for i in range(len(y_test)):
                label = y_test[i].item()
                cls_name = classes[label]
                test_class_total[label] += 1
                if predicted[i] == label:
                    test_class_correct[label] += 1

                subcat = get_subcategory(file_paths[i])
                subcat_total_per_class[cls_name][subcat] += 1
                if predicted[i] == label:
                    subcat_correct_per_class[cls_name][subcat] += 1

    test_acc = 100 * sum(test_class_correct) / sum(test_class_total)
    test_class_accs = [100 * c / t for c, t in zip(test_class_correct, test_class_total)]

    print("\n> Test Results:")
    print(f"  Overall: {test_acc:.2f}%")
    for cls, acc in zip(classes, test_class_accs):
        print(f"  {cls}: {acc:.2f}%")

    print("\n> Per Sub-Category (per class):")
    for cls in classes:
        print(f"  {cls}:")
        for subcat, total in subcat_total_per_class[cls].items():
            correct = subcat_correct_per_class[cls][subcat]
            acc = 100 * correct / total
            train_ratio = 100 * len(dataset_train) / (len(dataset_train) + total)
            print(f"    {subcat} ({train_ratio:.2f}/{100 - train_ratio:.2f}): {acc:.2f}%")
            subcat_summary[subcat].append(acc)
    
    saved.append(test_acc)
    good.append(test_class_accs[0])
    broken.append(test_class_accs[1])
    heavyload.append(test_class_accs[2])

# --------------------------------------------------------------------------------------------------------------------------------
# Final summary
# --------------------------------------------------------------------------------------------------------------------------------
print(f"\n{'='*128}")
print(f"FINAL RESULTS ({RUNS} runs)")
print(f"{'='*128}")

print(f"\nOverall: {np.mean(saved):.2f}% ± {np.std(saved):.2f}%")
print(f"  Range: {min(saved):.2f}% - {max(saved):.2f}%")

print("\nPer-Class Performance:")
print(f"  Good:      {np.mean(good):.2f}% ± {np.std(good):.2f}%")
print(f"  Broken:    {np.mean(broken):.2f}% ± {np.std(broken):.2f}%")
print(f"  Heavyload: {np.mean(heavyload):.2f}% ± {np.std(heavyload):.2f}%")

print("\nPer-Subcategory Average:")
for subcat, accs in subcat_summary.items():
    padding = " " * (14 - len(subcat))
    print(f"  {subcat}:{padding} {np.mean(accs):.2f}% ± {np.std(accs):.2f}%")
print(f"{'='*124}done")


    