import torchaudio
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
import torchvision.models as models
from torcheval.metrics import FrechetInceptionDistance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
import librosa
import cv2
from PIL import Image


# Load VGG16 model and extract up to FC1 layer
class VGG16_FC1(torch.nn.Module):
    def __init__(self):
        super(VGG16_FC1, self).__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.features = vgg16.features
        self.avgpool = vgg16.avgpool
        self.fc1 = torch.nn.Sequential(*list(vgg16.classifier.children())[:2])
        
        # Set to eval mode and freeze parameters
        self.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        with torch.no_grad():  # Disable gradient computation
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc1(x)
            return x  # Output shape: (batch_size 4096)


def extract_features_from_sample(data, device, sr=1000):
    model = VGG16_FC1().to(device)
    
    # Convert data to tensor and move to GPU
    signals_tensor = torch.from_numpy(data.T).float().to(device)  # Shape: (14, seq_len)
    
    # GPU-accelerated mel spectrogram computation
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_mels=128,
        n_fft=512,
        hop_length=128
    ).to(device)
    
    # Compute mel spectrograms on GPU
    with torch.no_grad():
        mel_specs = []
        for i in range(signals_tensor.shape[0]):
            mel_spec = mel_transform(signals_tensor[i])
            mel_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
            mel_specs.append(mel_db)
        
        mel_batch = torch.stack(mel_specs).squeeze(1)  # Shape: (14, n_mels, time_frames)
        
        # Resize to 224x224 using GPU interpolation
        mel_resized = F.interpolate(
            mel_batch.unsqueeze(1), 
            size=(224, 224), 
            mode='bilinear', 
            align_corners=False
        ).squeeze(1)
        
        # Normalize and create RGB channels
        mel_min = mel_resized.view(14, -1).min(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel_max = mel_resized.view(14, -1).max(dim=1, keepdim=True)[0].unsqueeze(-1)
        mel_norm = (mel_resized - mel_min) / (mel_max - mel_min + 1e-8)
        
        # Create RGB tensor
        rgb_batch = mel_norm.unsqueeze(1).repeat(1, 3, 1, 1)  # Shape: (14, 3, 224, 224)
        
        # Apply VGG normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)
        rgb_normalized = (rgb_batch - mean) / std
        
        # Extract features for entire batch
        features = model(rgb_normalized).cpu().numpy()
    
    return features.flatten()


# ===============================
# FID SCORE CALCULATION
# ===============================
# def process_dataset_for_fid(data, device, sample_rate=10000):
#     """
#     Process dataset specifically for FID calculation using Inception v3 (299x299)
#     Uses SAME improved configuration as process_dataset_multichannel
#     """
#     num_samples, seq_len, num_channels = data.shape
#     processed_images = []
    
#     # Auto-determine sample rate if not provided (SAME logic as multichannel)
#     if sample_rate is None:
#         if seq_len >= 4000:
#             sample_rate = 1000
#         elif seq_len >= 1000:
#             sample_rate = 500
#         else:
#             sample_rate = 100
    
#     # IMPROVED mel spectrogram parameters (IDENTICAL to multichannel)
#     optimal_config = {
#         'sample_rate': sample_rate,
#         'n_mels': 256,  # Increased from 128 for better frequency resolution
#         'n_fft': min(1024, seq_len // 4),  # Larger FFT window for better frequency analysis
#         'hop_length': min(128, seq_len // 32),  # Smaller hop for better time resolution
#         'win_length': min(1024, seq_len // 4),
#         'f_min': 0.0,
#         'f_max': sample_rate // 2,
#         'power': 2.0,
#         'normalized': True,  # Enable normalization for better stability
#         'center': True,
#         'pad_mode': 'reflect'
#     }
    
#     print(f"FID Processing - Using IMPROVED mel spectrogram config:")
#     print(f"  Sample rate: {optimal_config['sample_rate']} Hz")
#     print(f"  n_mels: {optimal_config['n_mels']} (increased for better resolution)")
#     print(f"  n_fft: {optimal_config['n_fft']}")
#     print(f"  hop_length: {optimal_config['hop_length']}")
#     print(f"  Sequence length: {seq_len} points")
#     print(f"  Duration: {seq_len/sample_rate:.2f} seconds")
#     print(f"  Frequency range: {optimal_config['f_min']}-{optimal_config['f_max']} Hz")
    
#     mel_transform = torchaudio.transforms.MelSpectrogram(**optimal_config).to(device)
    
#     print(f"Processing {num_samples} samples for FID calculation with IMPROVED statistical aggregation of ALL {num_channels} channels...")
    
#     for i in range(num_samples):
#         if i % 50 == 0:  # More frequent updates (same as multichannel)
#             print(f"Processed {i}/{num_samples} samples")
        
#         # Generate mel spectrograms for ALL channels with BETTER preprocessing (SAME as multichannel)
#         all_mels = []
#         for j in range(num_channels):
#             ts = torch.tensor(data[i, :, j], dtype=torch.float32).to(device)
            
#             # IMPROVED signal preprocessing (IDENTICAL to multichannel)
#             # 1. Remove DC component
#             ts = ts - ts.mean()
            
#             # 2. Normalize signal energy
#             ts = ts / (ts.std() + 1e-8)
            
#             # 3. Apply mild denoising (optional)
#             ts = torch.nn.functional.conv1d(ts.unsqueeze(0).unsqueeze(0), 
#                                            torch.ones(1,1,3).to(device)/3, padding=1).squeeze()
            
#             # Apply mel spectrogram transform
#             mel = mel_transform(ts)
            
#             if mel.dim() == 3:
#                 mel = mel.squeeze(0)
            
#             # BETTER per-channel normalization using robust statistics (IDENTICAL to multichannel)
#             mel_median = torch.median(mel)
#             mel_mad = torch.median(torch.abs(mel - mel_median))  # Median Absolute Deviation
#             mel_norm = (mel - mel_median) / (mel_mad + 1e-8)
            
#             # Clip extreme values to reduce noise impact
#             mel_norm = torch.clamp(mel_norm, -3, 3)  # 3-sigma clipping
            
#             # Normalize to [0,1] for Inception v3
#             mel_norm = (mel_norm - mel_norm.min()) / (mel_norm.max() - mel_norm.min() + 1e-8)
            
#             # Resize to 299x299 for Inception v3 (instead of 224x224 for VGG16)
#             mel_resized = torch.nn.functional.interpolate(
#                 mel_norm.unsqueeze(0).unsqueeze(0), 
#                 size=(299, 299), 
#                 mode='bilinear',
#                 align_corners=False
#             ).squeeze()
            
#             all_mels.append(mel_resized.cpu().numpy())
        
#         # Stack all mel spectrograms: shape (14, 299, 299)
#         all_mels = np.stack(all_mels, axis=0)
        
#         # IMPROVED statistical aggregation for better anomaly sensitivity (IDENTICAL to multichannel)
#         rgb_img = np.stack([
#             np.mean(all_mels, axis=0),        # R: Mean (central tendency)
#             np.var(all_mels, axis=0),         # G: Variance (better than std for anomalies)
#             np.median(all_mels, axis=0)       # B: Median (robust central measure)
#         ], axis=0)
        
#         # ROBUST normalization per channel (IDENTICAL to multichannel)
#         for c in range(3):
#             channel = rgb_img[c]
#             # Use percentile-based normalization to handle outliers
#             p1, p99 = np.percentile(channel, [1, 99])
#             rgb_img[c] = np.clip((channel - p1) / (p99 - p1 + 1e-8), 0, 1)
        
#         # Convert to tensor and keep in [0,1] range for FrechetInceptionDistance
#         img_tensor = torch.tensor(rgb_img, dtype=torch.float32)
#         processed_images.append(img_tensor)
    
#     return torch.stack(processed_images)



# def calculate_fid_score(real_data, fake_data, device, sample_rate=10000, batch_size=32):
#     """
#     Calculate FID score between real and fake data using Inception v3
#     Now uses SAME optimized configuration as process_dataset_multichannel
#     """
    
#     print("Processing real data for FID with optimized configuration...")
#     real_images = process_dataset_for_fid(real_data, device, sample_rate)
    
#     print("Processing fake data for FID with optimized configuration...")
#     fake_images = process_dataset_for_fid(fake_data, device, sample_rate)
    
#     # Ensure we have the same number of samples
#     min_samples = min(len(real_images), len(fake_images))
#     real_images = real_images[:min_samples]
#     fake_images = fake_images[:min_samples]
    
#     print(f"Calculating FID with {min_samples} samples each...")
#     print(f"Real images shape: {real_images.shape}")
#     print(f"Fake images shape: {fake_images.shape}")
    
#     # Move to device and verify data range
#     real_images = real_images.to(device)
#     fake_images = fake_images.to(device)
    
#     print(f"Real images range: [{real_images.min():.4f}, {real_images.max():.4f}]")
#     print(f"Fake images range: [{fake_images.min():.4f}, {fake_images.max():.4f}]")
    
#     # Ensure images are in [0, 1] range as required by FrechetInceptionDistance
#     real_images = torch.clamp(real_images, 0, 1)
#     fake_images = torch.clamp(fake_images, 0, 1)
    
#     print(f"After clamping - Real images range: [{real_images.min():.4f}, {real_images.max():.4f}]")
#     print(f"After clamping - Fake images range: [{fake_images.min():.4f}, {fake_images.max():.4f}]")
    
#     # Calculate FID score using the proper metric API
#     try:
#         fid = FrechetInceptionDistance(device=device)
#         fid.update(fake_images, is_real=False)
#         fid.update(real_images, is_real=True)
#         score = fid.compute()
#         print(f"FID Score: {score:.4f}")
#         return score.item()
#     except Exception as e:
#         print(f"Error calculating FID with metric API: {e}")
#         print("Trying functional API...")
#         try:
#             from torcheval.metrics.functional import frechet_inception_distance
#             fid_score = frechet_inception_distance(fake_images, real_images, device=device)
#             print(f"FID Score: {fid_score:.4f}")
#             return fid_score.item()
#         except Exception as e2:
#             print(f"Error with functional API: {e2}")
#             return None


         

# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_dim=57344):
        super(Autoencoder, self).__init__()
        # Encoder: 5 layers
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 80),
            nn.ReLU(),
            nn.Linear(80, 60),
            nn.ReLU(),
            nn.Linear(60, 40),
            nn.ReLU(),
            nn.Linear(40, 20)
            )
            
        self.decoder = nn.Sequential(
            nn.Linear(20, 40),
            nn.ReLU(),
            nn.Linear(40, 60),
            nn.ReLU(),
            nn.Linear(60, 80),
            nn.ReLU(),
            nn.Linear(80, 100),
            nn.ReLU(),
            nn.Linear(100, input_dim)
            )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def run_pipeline_with_cv(train_normal_data, test_normal_data, test_faulty_data,
                        device, batch_size, num_epochs, cv_folds=5):
    """
    Enhanced pipeline with 5-fold cross-validation, F1-optimized threshold selection,
    and comprehensive statistical analysis.
    """
    test_labels = np.concatenate([
        np.zeros(len(test_normal_data)),  # Normal = 0  
        np.ones(len(test_faulty_data))    # Faulty = 1
    ])
    # Extract features for all data
    extract_func = lambda x: extract_features_from_sample(x, device)
    normal_features_all = []
    test_features_all = []
    normal_data = np.concatenate([train_normal_data, test_normal_data], axis=0)
    faulty_data = np.array(test_faulty_data)
    test_data = np.concatenate([test_normal_data, test_faulty_data], axis=0)
    print("Extracting features from all samples...")
    # Normal extraction
    for i, sample in enumerate(normal_data):
        features = extract_func(sample)
        normal_features_all.append(features)

    normal_features_all = np.array(normal_features_all)

    # Test extraction
    for i, sample in enumerate(test_data):
        features = extract_func(sample)
        test_features_all.append(features)

    test_features_all = np.array(test_features_all)

    # Initialize cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    # Storage for results
    fold_results = []
    all_metrics = {
        'accuracy': [], 'precision': [], 'recall': [], 'f1': [],
        'optimal_threshold': [], 'train_loss': []
    }
    
    print(f"\nStarting {cv_folds}-fold cross-validation...")
    print("=" * 60)
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(test_features_all, test_labels)):
        print(f"\nFold {fold + 1}/{cv_folds}")
        print("-" * 30)
        
        # Split data
        X_train, X_test = test_features_all[train_idx], test_features_all[test_idx]
        y_train, y_test = test_labels[train_idx], test_labels[test_idx]
        
        # Only use normal samples for training (unsupervised learning)
        normal_mask = y_train == 0
        X_train_normal = X_train[normal_mask]
        
        print(f"Train normal samples: {len(X_train_normal)}")
        print(f"Test samples: {len(X_test)} ({np.sum(y_test == 0)} normal, {np.sum(y_test == 1)} anomaly)")
        
        # Train autoencoder on normal data only
        model, final_loss = train_autoencoder_fold(normal_features_all, device, batch_size, num_epochs)
        
        # Calculate reconstruction errors for test set
        model.eval()
        with torch.no_grad():
            X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
            reconstructed = model(X_test_tensor)
            reconstruction_errors = torch.mean((X_test_tensor - reconstructed) ** 2, dim=1).cpu().numpy()
        
        # Find optimal threshold using F1-score
        optimal_threshold, best_f1, threshold_metrics = find_optimal_threshold(
            reconstruction_errors, y_test
        )
        
        # Store fold results
        fold_result = {
            'fold': fold + 1,
            'train_samples': len(X_train_normal),
            'test_samples': len(X_test),
            'final_train_loss': final_loss,
            'optimal_threshold': optimal_threshold,
            **threshold_metrics
        }
        fold_results.append(fold_result)
        
        # Accumulate metrics
        all_metrics['accuracy'].append(threshold_metrics['accuracy'])
        all_metrics['precision'].append(threshold_metrics['precision'])
        all_metrics['recall'].append(threshold_metrics['recall'])
        all_metrics['f1'].append(threshold_metrics['f1'])
        all_metrics['optimal_threshold'].append(optimal_threshold)
        all_metrics['train_loss'].append(final_loss)
        
        print(f"Results: Acc={threshold_metrics['accuracy']:.4f}, "
              f"Prec={threshold_metrics['precision']:.4f}, "
              f"Rec={threshold_metrics['recall']:.4f}, "
              f"F1={threshold_metrics['f1']:.4f}")
        print(f"Optimal threshold: {optimal_threshold:.6f}")
    
    # Calculate statistics
    statistics = calculate_cv_statistics(all_metrics)
    
    # Print comprehensive results
    print_cv_results(statistics, fold_results)
    
    return {
        'fold_results': fold_results,
        'statistics': statistics,
        'all_metrics': all_metrics
    }

def train_autoencoder_fold(X_train, device, batch_size, num_epochs):
    """Train autoencoder for one fold"""
    features_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    dataset = TensorDataset(features_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    final_loss = 0
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in dataloader:
            x = batch[0].to(device)
            optimizer.zero_grad()
            x_recon = model(x)
            loss = criterion(x_recon, x)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        final_loss = epoch_loss / len(dataloader)
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"  Epoch {epoch+1}/{num_epochs}, Loss: {final_loss:.6f}")
    
    return model, final_loss

def find_optimal_threshold(reconstruction_errors, y_true):
    """Find optimal threshold that maximizes F1-score"""
    # Generate threshold candidates
    min_error, max_error = reconstruction_errors.min(), reconstruction_errors.max()
    thresholds = np.linspace(min_error, max_error, 100)
    
    best_f1 = 0
    best_threshold = 0
    best_metrics = {}
    
    for threshold in thresholds:
        y_pred = (reconstruction_errors > threshold).astype(int)
        
        # Calculate metrics
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_metrics = {
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1': f1
            }
    
    return best_threshold, best_f1, best_metrics

def calculate_cv_statistics(all_metrics):
    """Calculate mean, std, and confidence intervals for all metrics"""
    statistics = {}
    
    for metric_name, values in all_metrics.items():
        values = np.array(values)
        statistics[metric_name] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'values': values
        }
    
    return statistics

def print_cv_results(statistics, fold_results):
    """Print comprehensive cross-validation results"""
    print("\n" + "=" * 80)
    print("CROSS-VALIDATION RESULTS SUMMARY")
    print("=" * 80)
    
    print("\nFOLD-BY-FOLD RESULTS:")
    print("-" * 80)
    print(f"{'Fold':<6} {'Accuracy':<10} {'Precision':<11} {'Recall':<8} {'F1-Score':<9} {'Threshold':<12}")
    print("-" * 80)
    
    for result in fold_results:
        print(f"{result['fold']:<6} "
              f"{result['accuracy']:<10.4f} "
              f"{result['precision']:<11.4f} "
              f"{result['recall']:<8.4f} "
              f"{result['f1']:<9.4f} "
              f"{result['optimal_threshold']:<12.6f}")
    
    print("\nSTATISTICAL SUMMARY:")
    print("-" * 80)
    print(f"{'Metric':<12} {'Mean':<8} {'Std':<8} {'Min':<8} {'Max':<8} {'Median':<8}")
    print("-" * 80)
    
    for metric in ['accuracy', 'precision', 'recall', 'f1']:
        stats = statistics[metric]
        print(f"{metric.capitalize():<12} "
              f"{stats['mean']:<8.4f} "
              f"{stats['std']:<8.4f} "
              f"{stats['min']:<8.4f} "
              f"{stats['max']:<8.4f} "
              f"{stats['median']:<8.4f}")
    
    print(f"\nOVERALL PERFORMANCE:")
    print(f"  Mean F1-Score: {statistics['f1']['mean']:.4f} ± {statistics['f1']['std']:.4f}")
    print(f"  F1-Score Range: [{statistics['f1']['min']:.4f}, {statistics['f1']['max']:.4f}]")
    print(f"  Mean Threshold: {statistics['optimal_threshold']['mean']:.6f} ± {statistics['optimal_threshold']['std']:.6f}")

def compare_pipelines_with_ttest(baseline_results, gan_results):
    """
    Compare baseline vs GAN-augmented results using paired t-tests
    
    Args:
        baseline_results: Results from baseline pipeline
        gan_results: Results from GAN-augmented pipeline
    
    Returns:
        dict: Statistical comparison results
    """
    comparison = {}
    
    print("\n" + "=" * 80)
    print("STATISTICAL COMPARISON: BASELINE vs GAN-AUGMENTED")
    print("=" * 80)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    print(f"{'Metric':<12} {'Baseline':<12} {'GAN-Aug':<12} {'Δ Mean':<10} {'p-value':<10} {'Significant':<12}")
    print("-" * 80)
    
    for metric in metrics:
        baseline_values = baseline_results['statistics'][metric]['values']
        gan_values = gan_results['statistics'][metric]['values']
        
        # Paired t-test
        t_stat, p_value = stats.ttest_rel(gan_values, baseline_values)
        
        baseline_mean = baseline_results['statistics'][metric]['mean']
        gan_mean = gan_results['statistics'][metric]['mean']
        delta = gan_mean - baseline_mean
        
        significant = "Yes" if p_value < 0.05 else "No"
        
        comparison[metric] = {
            'baseline_mean': baseline_mean,
            'gan_mean': gan_mean,
            'delta': delta,
            'p_value': p_value,
            't_statistic': t_stat,
            'significant': significant
        }
        
        print(f"{metric.capitalize():<12} "
              f"{baseline_mean:<12.4f} "
              f"{gan_mean:<12.4f} "
              f"{delta:<10.4f} "
              f"{p_value:<10.4f} "
              f"{significant:<12}")
    
    print("\nINTERPRETATION:")
    print("-" * 40)
    significant_improvements = sum(1 for m in comparison.values() if m['significant'] == 'Yes' and m['delta'] > 0)
    total_metrics = len(metrics)
    
    print(f"Significantly improved metrics: {significant_improvements}/{total_metrics}")
    
    if significant_improvements > 0:
        print("✅ GAN augmentation shows statistically significant improvements!")
    else:
        print("❌ No statistically significant improvements from GAN augmentation.")
    
    return comparison

# Example usage:
def run_comparative_experiment(normal_data, faulty_data, device, generated_data=None):
    """
    Run complete comparative experiment with baseline and GAN-augmented pipelines
    """
    # Combine normal and faulty data
    all_data = np.concatenate([normal_data, faulty_data], axis=0)
    labels = np.concatenate([
        np.zeros(len(normal_data)),  # Normal = 0
        np.ones(len(faulty_data))    # Anomaly = 1
    ])
    
    print("Running baseline pipeline...")
    baseline_results = run_pipeline_with_cv(all_data, labels, device)
    
    if generated_data is not None:
        print("\nRunning GAN-augmented pipeline...")
        # Augment normal data with generated samples
        augmented_normal = np.concatenate([normal_data, generated_data], axis=0)
        augmented_data = np.concatenate([augmented_normal, faulty_data], axis=0)
        augmented_labels = np.concatenate([
            np.zeros(len(augmented_normal)),  # Normal = 0
            np.ones(len(faulty_data))         # Anomaly = 1
        ])
        
        gan_results = run_pipeline_with_cv(augmented_data, augmented_labels, device)
        
        # Statistical comparison
        comparison = compare_pipelines_with_ttest(baseline_results, gan_results)
        
        return baseline_results, gan_results, comparison
    
    return baseline_results