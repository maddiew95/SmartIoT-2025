import torchaudio
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vgg16, VGG16_Weights
from torcheval.metrics import FrechetInceptionDistance
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import KFold
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt


# Alternative: Multi-channel processing
def process_dataset_multichannel(data, device, sample_rate=1000):
    """
    Process multiple channels together to capture cross-channel relationships
    """
    num_samples, _, num_channels = data.shape
    features = np.zeros((num_samples, 4096))  # Single feature vector per sample
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=128,
        n_fft=512,
        hop_length=256,
        win_length=512
    ).to(device)
    
    model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    model.classifier = model.classifier[:-3]
    model.eval()
    
    print(f"Processing {num_samples} samples with multi-channel approach...")
    for i in range(num_samples):
        # if i % 100 == 0:
        #     print(f"Processed {i}/{num_samples} samples")
        
        # Combine multiple channels into RGB image
        channel_spectrograms = []
        for j in range(min(3, num_channels)):  # Use first 3 channels as RGB
            ts = torch.tensor(data[i, :, j], dtype=torch.float32).to(device)
            mel = mel_transform(ts)
            
            # Normalize each channel spectrogram
            mel_norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            mel_resized = torch.nn.functional.interpolate(
                mel_norm.unsqueeze(0).unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear'
            ).squeeze()
            channel_spectrograms.append(mel_resized.cpu().numpy())
        
        # Stack as RGB image
        if len(channel_spectrograms) == 1:
            rgb_img = np.stack([channel_spectrograms[0]] * 3, axis=0)
        elif len(channel_spectrograms) == 2:
            rgb_img = np.stack([channel_spectrograms[0], channel_spectrograms[1], channel_spectrograms[0]], axis=0)
        else:
            rgb_img = np.stack(channel_spectrograms[:3], axis=0)
        
        img_tensor = torch.tensor(rgb_img, dtype=torch.float32).unsqueeze(0).to(device)
        
        with torch.no_grad():
            feat = model(img_tensor)
        features[i, :] = feat.squeeze().cpu().numpy()
    
    return features

# ===============================
# FID SCORE CALCULATION
# ===============================

def process_dataset_for_fid(data, device, sample_rate=1000):
    """
    Process dataset specifically for FID calculation using Inception v3 (299x299)
    """
 
    
    num_samples, _, num_channels = data.shape
    processed_images = []
    
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_mels=128,
        n_fft=512,
        hop_length=256,
        win_length=512
    ).to(device)
    
    print(f"Processing {num_samples} samples for FID calculation (299x299)...")
    for i in range(num_samples):
        if i % 100 == 0:
            print(f"Processed {i}/{num_samples} samples")
        
        # Combine multiple channels into RGB image
        channel_spectrograms = []
        for j in range(min(3, num_channels)):  # Use first 3 channels as RGB
            ts = torch.tensor(data[i, :, j], dtype=torch.float32).to(device)
            mel = mel_transform(ts)
            
            # Normalize each channel spectrogram
            mel_norm = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)
            # Resize to 299x299 for Inception v3
            mel_resized = torch.nn.functional.interpolate(
                mel_norm.unsqueeze(0).unsqueeze(0), 
                size=(299, 299), 
                mode='bilinear'
            ).squeeze()
            channel_spectrograms.append(mel_resized.cpu().numpy())
        
        # Stack as RGB image
        if len(channel_spectrograms) == 1:
            rgb_img = np.stack([channel_spectrograms[0]] * 3, axis=0)
        elif len(channel_spectrograms) == 2:
            rgb_img = np.stack([channel_spectrograms[0], channel_spectrograms[1], channel_spectrograms[0]], axis=0)
        else:
            rgb_img = np.stack(channel_spectrograms[:3], axis=0)
        
        # Convert to tensor and keep in [0,1] range for FrechetInceptionDistance
        img_tensor = torch.tensor(rgb_img, dtype=torch.float32)
        # img_tensor is already in [0,1] range from mel_norm, no need to convert to [-1,1]
        processed_images.append(img_tensor)
    
    return torch.stack(processed_images)

def calculate_fid_score(real_data, fake_data, device, sample_rate=1000, batch_size=32):
    """
    Calculate FID score between real and fake data using Inception v3
    """
    
    print("Processing real data for FID...")
    real_images = process_dataset_for_fid(real_data, device, sample_rate)
    
    print("Processing fake data for FID...")
    fake_images = process_dataset_for_fid(fake_data, device, sample_rate)
    
    # Ensure we have the same number of samples
    min_samples = min(len(real_images), len(fake_images))
    real_images = real_images[:min_samples]
    fake_images = fake_images[:min_samples]
    
    print(f"Calculating FID with {min_samples} samples each...")
    print(f"Real images shape: {real_images.shape}")
    print(f"Fake images shape: {fake_images.shape}")
    
    # Move to device and verify data range
    real_images = real_images.to(device)
    fake_images = fake_images.to(device)
    
    print(f"Real images range: [{real_images.min():.4f}, {real_images.max():.4f}]")
    print(f"Fake images range: [{fake_images.min():.4f}, {fake_images.max():.4f}]")
    
    # Ensure images are in [0, 1] range as required by FrechetInceptionDistance
    real_images = torch.clamp(real_images, 0, 1)
    fake_images = torch.clamp(fake_images, 0, 1)
    
    print(f"After clamping - Real images range: [{real_images.min():.4f}, {real_images.max():.4f}]")
    print(f"After clamping - Fake images range: [{fake_images.min():.4f}, {fake_images.max():.4f}]")
    
    # Calculate FID score using the proper metric API
    try:
        fid = FrechetInceptionDistance(device=device)
        fid.update(fake_images, is_real=False)
        fid.update(real_images, is_real=True)
        score = fid.compute()
        print(f"FID Score: {score:.4f}")
        return score.item()
    except Exception as e:
        print(f"Error calculating FID with metric API: {e}")
        print("Trying functional API...")
        try:
            from torcheval.metrics.functional import frechet_inception_distance
            fid_score = frechet_inception_distance(fake_images, real_images, device=device)
            print(f"FID Score: {fid_score:.4f}")
            return fid_score.item()
        except Exception as e2:
            print(f"Error with functional API: {e2}")
            print("Trying batch-wise calculation...")
            return calculate_fid_batch_wise(real_images, fake_images, batch_size)

def calculate_fid_batch_wise(real_images, fake_images, device, batch_size=32):
    """
    Calculate FID in batches if memory is limited
    """    
    n_samples = len(real_images)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    real_features_list = []
    fake_features_list = []
    
    # Load Inception v3 for feature extraction
    from torchvision.models import inception_v3
    inception_model = inception_v3(pretrained=True, transform_input=False).to(device)
    inception_model.eval()
    
    print("Extracting features batch-wise...")
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, n_samples)
            
            real_batch = real_images[start_idx:end_idx]
            fake_batch = fake_images[start_idx:end_idx]
            
            # Extract features using Inception v3
            real_features = inception_model(real_batch)
            fake_features = inception_model(fake_batch)
            
            real_features_list.append(real_features.cpu())
            fake_features_list.append(fake_features.cpu())
            
            if i % 10 == 0:
                print(f"Processed batch {i+1}/{n_batches}")
    
    # Concatenate all features
    real_features = torch.cat(real_features_list, dim=0).to(device)
    fake_features = torch.cat(fake_features_list, dim=0).to(device)
    
    # Calculate FID using functional API with extracted features
    fid_score = FrechetInceptionDistance(fake_features, real_features, device=device)
    print(f"FID Score (batch-wise): {fid_score:.4f}")
    return fid_score.item()

def calculate_fid_simple(real_data, generated_data, device, sample_rate=1000, max_samples=200):
    """
    Simplified FID calculation with better error handling
    """
    print("=" * 50)
    print("SIMPLIFIED FID CALCULATION")
    print("=" * 50)
    
    # Limit samples for memory efficiency
    n_samples = min(len(real_data), len(generated_data), max_samples)
    real_subset = real_data[:n_samples]
    gen_subset = generated_data[:n_samples]
    
    print(f"Using {n_samples} samples for FID calculation")
    print(f"Real data shape: {real_subset.shape}")
    print(f"Generated data shape: {gen_subset.shape}")
    
    try:
        # Process mel spectrograms
        print("Processing mel spectrograms...")
        real_mels = process_dataset_for_fid(real_subset, sample_rate)
        gen_mels = process_dataset_for_fid(gen_subset, sample_rate)
        
        # Move to device and ensure [0,1] range
        real_mels = torch.clamp(real_mels.to(device), 0, 1)
        gen_mels = torch.clamp(gen_mels.to(device), 0, 1)
        
        print(f"Processed real mels shape: {real_mels.shape}")
        print(f"Processed gen mels shape: {gen_mels.shape}")
        print(f"Real mels range: [{real_mels.min():.4f}, {real_mels.max():.4f}]")
        print(f"Gen mels range: [{gen_mels.min():.4f}, {gen_mels.max():.4f}]")
        
        # Calculate FID using metric class
        print("Calculating FID score...")
        fid_metric = FrechetInceptionDistance(device=device)
        fid_metric.update(gen_mels, is_real=False)
        fid_metric.update(real_mels, is_real=True)
        score = fid_metric.compute()
        
        print(f"FID Score: {score:.4f}")
        return score.item()
        
    except Exception as e:
        print(f"Error in simplified FID calculation: {e}")
        print("Trying with smaller batch size...")
        try:
            # Try with even smaller batch
            small_n = min(50, n_samples)
            real_small = real_data[:small_n]
            gen_small = generated_data[:small_n]
            
            real_mels = process_dataset_for_fid(real_small, sample_rate)
            gen_mels = process_dataset_for_fid(gen_small, sample_rate)
            
            real_mels = torch.clamp(real_mels.to(device), 0, 1)
            gen_mels = torch.clamp(gen_mels.to(device), 0, 1)
            
            fid_metric = FrechetInceptionDistance(device=device)
            fid_metric.update(gen_mels, is_real=False)
            fid_metric.update(real_mels, is_real=True)
            score = fid_metric.compute()
            
            print(f"FID Score (small batch): {score:.4f}")
            return score.item()
            
        except Exception as e2:
            print(f"Final error: {e2}")
            return None


# Autoencoder model
class Autoencoder(nn.Module):
    def __init__(self, input_size=4096):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 64), 
            nn.Tanh(),
            nn.Linear(64, 32), 
            nn.Tanh(),
            nn.Linear(32, 16), 
            nn.Tanh(),
            nn.Linear(16, 8), 
            nn.Tanh(),
            nn.Linear(8, 4), 
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.Tanh(),
            nn.Linear(8, 16), 
            nn.Tanh(),
            nn.Linear(16, 32), 
            nn.Tanh(),
            nn.Linear(32, 64), 
            nn.Tanh(),
            nn.Linear(64, input_size), 
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# Train autoencoder
def train_autoencoder(features, device, epochs=20, batch_size=128):
    x = torch.tensor(features.reshape(-1, 4096), dtype=torch.float32).to(device)
    loader = DataLoader(TensorDataset(x), batch_size=batch_size, shuffle=True)
    model = Autoencoder().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Add weight decay
    criterion = nn.MSELoss()  # Try MSE instead of L1

    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            inputs = batch[0]
            # Add noise for denoising autoencoder
            noisy_inputs = inputs + 0.1 * torch.randn_like(inputs)
            outputs = model(noisy_inputs)
            loss = criterion(outputs, inputs)  # Reconstruct clean from noisy
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        # print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.6f}")
    return model

# Compute reconstruction errors
def compute_reconstruction_loss(model, data, add_noise=True):
    """
    Compute reconstruction loss per sample (not per segment)
    data: shape (n_samples, n_channels, 4096)
    """
    model.eval()
    n_samples, n_channels, n_features = data.shape
    sample_errors = []
    
    # Flatten to (n_samples*n_channels, 4096) for batch processing
    x = torch.tensor(data.reshape(-1, n_features), dtype=torch.float32).to(next(model.parameters()).device)
    loader = DataLoader(TensorDataset(x), batch_size=64)
    
    all_errors = []
    criterion = torch.nn.MSELoss(reduction='none')
    
    with torch.no_grad():
        for batch in loader:
            inputs = batch[0]
            
            if add_noise:
                noisy_inputs = inputs + 0.1 * torch.randn_like(inputs)
                outputs = model(noisy_inputs)
            else:
                outputs = model(inputs)
            
            # Per-segment reconstruction error
            segment_errors = criterion(outputs, inputs).mean(dim=1)
            all_errors.extend(segment_errors.cpu().numpy())
    
    # Reshape back to (n_samples, n_channels) and aggregate per sample
    all_errors = np.array(all_errors).reshape(n_samples, n_channels)
    sample_errors = all_errors.mean(axis=1)  # Average across channels per sample
    
    return sample_errors

# 2. Find best threshold based on F1 score
def find_best_threshold(errors, labels):
    best_f1 = 0
    best_threshold = 0
    for threshold in np.linspace(min(errors), max(errors), 100):
        preds = (errors > threshold).astype(int)
        f1 = f1_score(labels, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold, best_f1

def find_best_threshold_using_recall(errors, labels):
    best_rec = 0
    best_threshold = 0
    for threshold in np.linspace(min(errors), max(errors), 100):
        preds = (errors > threshold).astype(int)
        rec = recall_score(labels, preds)
        if rec > best_rec:
            best_rec = rec
            best_threshold = threshold
    return best_threshold, best_rec

def find_best_threshold_using_precision(errors, labels):
    best_prec = 0
    best_threshold = 0
    for threshold in np.linspace(min(errors), max(errors), 100):
        preds = (errors > threshold).astype(int)
        prec = precision_score(labels, preds)
        if prec > best_prec:
            best_prec = prec
            best_threshold = threshold
    return best_threshold, best_prec

def find_best_threshold_using_accuracy(errors, labels):
    """Find best threshold based on accuracy"""
    best_acc = 0
    best_threshold = 0
    for threshold in np.linspace(min(errors), max(errors), 100):
        preds = (errors > threshold).astype(int)
        acc = accuracy_score(labels, preds)
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
    return best_threshold, best_acc

def get_percentile_threshold(errors, percentile=95):
    """Get threshold based on percentile (e.g., 90th, 95th)"""
    return np.percentile(errors, percentile)

# def train_ocsvm_on_features(features, contamination=0.1):
#     """
#     Train One-Class SVM on autoencoder features (encoded representations)
#     features: encoded features from autoencoder encoder
#     """
#     from sklearn.svm import OneClassSVM
#     from sklearn.preprocessing import StandardScaler
    
#     # Standardize features
#     scaler = StandardScaler()
#     features_scaled = scaler.fit_transform(features)
    
#     # Train One-Class SVM
#     ocsvm = OneClassSVM(kernel='rbf', gamma='scale')
#     ocsvm.fit(features_scaled)
    
#     return ocsvm, scaler


def predict_with_binary_svm(model, binary_svm, scaler, X_data):
    """
    Get Binary SVM predictions using autoencoder encoded features
    """
    model.eval()
    device = next(model.parameters()).device
    
    # Get encoded features
    n_samples, n_channels, n_features = X_data.shape
    x = torch.tensor(X_data.reshape(-1, n_features), dtype=torch.float32).to(device)
    
    encoded_features = []
    with torch.no_grad():
        for i in range(0, len(x), 64):  # Process in batches
            batch = x[i:i+64]
            encoded = model.encoder(batch)
            encoded_features.append(encoded.cpu().numpy())
    
    encoded_features = np.vstack(encoded_features)
    # Reshape back and average across channels
    encoded_features = encoded_features.reshape(n_samples, n_channels, -1).mean(axis=1)
    
    # Scale and predict
    encoded_scaled = scaler.transform(encoded_features)
    predictions = binary_svm.predict(encoded_scaled)
    
    # predictions are already 0/1 (0=normal, 1=anomaly)
    return predictions.astype(int)


# def predict_with_ocsvm(model, ocsvm, scaler, X_data):
#     """
#     Get OCSVM predictions using autoencoder encoded features
#     """
#     model.eval()
#     device = next(model.parameters()).device
    
#     # Get encoded features
#     n_samples, n_channels, n_features = X_data.shape
#     x = torch.tensor(X_data.reshape(-1, n_features), dtype=torch.float32).to(device)
    
#     encoded_features = []
#     with torch.no_grad():
#         for i in range(0, len(x), 64):  # Process in batches
#             batch = x[i:i+64]
#             encoded = model.encoder(batch)
#             encoded_features.append(encoded.cpu().numpy())
    
#     encoded_features = np.vstack(encoded_features)
#     # Reshape back and average across channels
#     encoded_features = encoded_features.reshape(n_samples, n_channels, -1).mean(axis=1)
    
#     # Scale and predict
#     encoded_scaled = scaler.transform(encoded_features)
#     predictions = ocsvm.predict(encoded_scaled)
    
#     # Convert to anomaly labels (1 = anomaly, 0 = normal)
#     anomaly_preds = (predictions == -1).astype(int)
    
#     return anomaly_preds


def evaluate_on_test_with_threshold_search(model, threshold, X_test, y_test):
    """
    Legacy function - kept for backward compatibility
    X_test: shape (n_samples, 1, 4096) - already has channel dimension added
    y_test: shape (n_samples,)
    """
    # X_test already has shape (n_samples, 1, 4096) from your code
    # So we can directly compute reconstruction errors
    test_errors = compute_reconstruction_loss(model, X_test)
    
    # Predict using best threshold
    test_preds = (test_errors > threshold).astype(int)

    # Evaluate
    print("Evaluation on Test Set:")
    print("Accuracy =", accuracy_score(y_test, test_preds))
    print("Precision =", precision_score(y_test, test_preds))
    print("Recall =", recall_score(y_test, test_preds))
    print("F1 Score =", f1_score(y_test, test_preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, test_preds))


# ===============================
# COMPREHENSIVE CROSS-VALIDATION FRAMEWORK
# ===============================

from sklearn.model_selection import StratifiedKFold
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt

def run_comprehensive_cross_validation_experiment(normal_data, faulty_data, device, generated_data=None, 
                                                  n_splits=5, epochs=20, batch_size=128, gan_type="Unknown"):
    """
    Unified comprehensive cross-validation experiment for anomaly detection
    
    Args:
        normal_data: Normal samples for training/testing
        faulty_data: Faulty samples for training/testing
        generated_data: Optional generated samples for data augmentation
        n_splits: Number of cross-validation folds
        epochs: Training epochs for autoencoder
        batch_size: Batch size for training
        gan_type: Type of GAN used for generated data (for statistical comparison)

    Returns:
        aggregated_results: Averaged metrics across folds with std deviations
        fold_results: Individual fold results
        rankings: Method rankings by different criteria
        gan_comparison_results: Results for GAN vs baseline comparison
    """
    print(f"\n{'='*70}")
    print("COMPREHENSIVE ANOMALY DETECTION CROSS-VALIDATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Normal samples: {len(normal_data)}")
    print(f"Faulty samples: {len(faulty_data)}")
    if generated_data is not None:
        print(f"Generated samples: {len(generated_data)} (GAN Type: {gan_type})")
    print(f"Cross-validation folds: {n_splits}")
    
    # Combine all data for stratified splitting
    all_data = np.concatenate([normal_data, faulty_data], axis=0)
    normal_labels = np.zeros(len(normal_data))
    faulty_labels = np.ones(len(faulty_data))
    all_labels = np.concatenate([normal_labels, faulty_labels], axis=0)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize results storage
    methods = ['F1_Threshold', 'Accuracy_Threshold']
    fold_results = []
    baseline_results = []  # Results without GAN augmentation
    
    # Process each fold
    for fold, (train_idx, test_idx) in enumerate(skf.split(all_data, all_labels)):
        print(f"\n{'='*25} FOLD {fold+1}/{n_splits} {'='*25}")
        
        # Split data by fold indices
        train_data_fold = all_data[train_idx]
        train_labels_fold = all_labels[train_idx]
        test_data_fold = all_data[test_idx]
        test_labels_fold = all_labels[test_idx]
        
        # Separate normal and faulty in training set
        normal_train_mask = train_labels_fold == 0
        faulty_train_mask = train_labels_fold == 1
        train_normal_fold = train_data_fold[normal_train_mask]
        train_faulty_fold = train_data_fold[faulty_train_mask]
        
        print(f"Fold {fold+1} - Train normal: {len(train_normal_fold)}")
        print(f"Fold {fold+1} - Train faulty: {len(train_faulty_fold)}")
        print(f"Fold {fold+1} - Test: {len(test_data_fold)} ({np.sum(test_labels_fold==0)} normal, {np.sum(test_labels_fold==1)} faulty)")
        
        # Process data through feature extraction pipeline
        print("Processing data through feature extraction...")
        train_faulty_features = process_dataset_multichannel(train_faulty_fold, device)
        test_features = process_dataset_multichannel(test_data_fold, device)
        
        # Add channel dimension for autoencoder compatibility
        train_faulty_features = train_faulty_features[:, np.newaxis, :]
        test_features = test_features[:, np.newaxis, :]
        
        # ===== BASELINE: Train without GAN augmentation =====
        baseline_normal_features = process_dataset_multichannel(train_normal_fold, device)
        baseline_normal_features = baseline_normal_features[:, np.newaxis, :]
        
        print("Training baseline autoencoder (without GAN)...")
        baseline_model = train_autoencoder(baseline_normal_features.reshape(-1, 4096), device, epochs, batch_size)
        baseline_test_errors = compute_reconstruction_loss(baseline_model, test_features)
        
        # Evaluate baseline methods
        baseline_fold_result = {}
        
        # F1-optimized threshold (baseline)
        baseline_f1_threshold, _ = find_best_threshold(baseline_test_errors, test_labels_fold)
        baseline_f1_preds = (baseline_test_errors > baseline_f1_threshold).astype(int)
        baseline_fold_result['F1_Threshold'] = {
            'accuracy': accuracy_score(test_labels_fold, baseline_f1_preds),
            'precision': precision_score(test_labels_fold, baseline_f1_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, baseline_f1_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, baseline_f1_preds, zero_division=0),
            'threshold': baseline_f1_threshold
        }
        
        # Accuracy-optimized threshold (baseline)
        baseline_acc_threshold, _ = find_best_threshold_using_accuracy(baseline_test_errors, test_labels_fold)
        baseline_acc_preds = (baseline_test_errors > baseline_acc_threshold).astype(int)
        baseline_fold_result['Accuracy_Threshold'] = {
            'accuracy': accuracy_score(test_labels_fold, baseline_acc_preds),
            'precision': precision_score(test_labels_fold, baseline_acc_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, baseline_acc_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, baseline_acc_preds, zero_division=0),
            'threshold': baseline_acc_threshold
        }
        
        
        baseline_results.append(baseline_fold_result)
        
        # ===== GAN-AUGMENTED: Train with GAN augmentation (if available) =====
        if generated_data is not None:
            augmented_normal_data = np.concatenate([generated_data, train_normal_fold], axis=0)
            print(f"Fold {fold+1} - Augmented normal data: {len(augmented_normal_data)} samples")
        else:
            augmented_normal_data = train_normal_fold
            print(f"Fold {fold+1} - Using original normal data only")
        
        augmented_normal_features = process_dataset_multichannel(augmented_normal_data, device)
        augmented_normal_features = augmented_normal_features[:, np.newaxis, :]
        
        # Train autoencoder on augmented normal data
        print("Training GAN-augmented autoencoder...")
        model = train_autoencoder(augmented_normal_features.reshape(-1, 4096), device, epochs, batch_size)
        
        # Compute reconstruction errors on test set
        test_errors = compute_reconstruction_loss(model, test_features)
        
        # Evaluate all methods with GAN augmentation
        fold_result = {}
        
        # F1-optimized threshold (GAN-augmented)
        f1_threshold, _ = find_best_threshold(test_errors, test_labels_fold)
        f1_preds = (test_errors > f1_threshold).astype(int)
        fold_result['F1_Threshold'] = {
            'accuracy': accuracy_score(test_labels_fold, f1_preds),
            'precision': precision_score(test_labels_fold, f1_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, f1_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, f1_preds, zero_division=0),
            'threshold': f1_threshold
        }
        
        # Accuracy-optimized threshold (GAN-augmented)
        acc_threshold, _ = find_best_threshold_using_accuracy(test_errors, test_labels_fold)
        acc_preds = (test_errors > acc_threshold).astype(int)
        fold_result['Accuracy_Threshold'] = {
            'accuracy': accuracy_score(test_labels_fold, acc_preds),
            'precision': precision_score(test_labels_fold, acc_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, acc_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, acc_preds, zero_division=0),
            'threshold': acc_threshold
        }

        # Print fold results comparison
        print(f"\nFold {fold+1} Results Comparison:")
        print("-" * 80)
        print(f"{'Method':<18} | {'Metric':<8} | {'Baseline':<10} | {'GAN-Aug':<10} | {'Improvement':<12}")
        print("-" * 80)
        for method in methods:
            for metric in ['accuracy', 'precision', 'recall', 'f1']:
                baseline_val = baseline_fold_result[method][metric]
                gan_val = fold_result[method][metric]
                improvement = gan_val - baseline_val
                improvement_str = f"{improvement:+.4f}"
                
                print(f"{method:<18} | {metric:<8} | {baseline_val:<10.4f} | {gan_val:<10.4f} | {improvement_str:<12}")
        
        fold_results.append(fold_result)
    
    # Aggregate results across folds
    aggregated_results = aggregate_results_with_std(fold_results, methods)
    baseline_aggregated = aggregate_results_with_std(baseline_results, methods, label="Baseline")
    
    # Compare GAN vs Baseline with statistical tests
    gan_comparison_results = compare_gan_vs_baseline(fold_results, baseline_results, methods, gan_type)
    
    # Rank methods
    rankings = rank_methods_with_std(aggregated_results, methods)
    
    # Create enhanced visualizations
    visualize_results_with_comparison(aggregated_results, baseline_aggregated, fold_results, baseline_results, methods, gan_type)
    
    # Provide enhanced recommendations
    provide_enhanced_recommendations(aggregated_results, baseline_aggregated, rankings, methods, gan_comparison_results, gan_type)
    
    return aggregated_results, fold_results, rankings, gan_comparison_results

def aggregate_results_with_std(fold_results, methods, label="GAN-Augmented"):
    """Aggregate results across all folds and compute statistics with enhanced std reporting"""
    print(f"\n{'='*70}")
    print(f"RESULTS AGGREGATION & STATISTICAL ANALYSIS ({label})")
    print(f"{'='*70}")
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Aggregate results
    aggregated_results = {}
    for method in methods:
        aggregated_results[method] = {}
        for metric in metrics:
            values = [fold_result[method][metric] for fold_result in fold_results]
            aggregated_results[method][metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values
            }
    
    # Print detailed results with std deviation
    print(f"\nDETAILED RESULTS SUMMARY ({label}):")
    print("-" * 90)
    print(f"{'Method':<18} | {'Accuracy':<15} | {'Precision':<15} | {'Recall':<15} | {'F1-Score':<15}")
    print("-" * 90)
    
    for method in methods:
        acc_stats = aggregated_results[method]['accuracy']
        prec_stats = aggregated_results[method]['precision']
        rec_stats = aggregated_results[method]['recall']
        f1_stats = aggregated_results[method]['f1']
        
        print(f"{method:<18} | {acc_stats['mean']:.4f}±{acc_stats['std']:.4f} | "
              f"{prec_stats['mean']:.4f}±{prec_stats['std']:.4f} | "
              f"{rec_stats['mean']:.4f}±{rec_stats['std']:.4f} | "
              f"{f1_stats['mean']:.4f}±{f1_stats['std']:.4f}")
    
    # Additional statistical summary
    print(f"\nSTATISTICAL SUMMARY ({label}):")
    print("-" * 80)
    for method in methods:
        print(f"\n{method}:")
        for metric in metrics:
            stats = aggregated_results[method][metric]
            print(f"  {metric.capitalize():<10}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                  f"[{stats['min']:.4f}, {stats['max']:.4f}] (median: {stats['median']:.4f})")
    
    return aggregated_results

def compare_gan_vs_baseline(gan_results, baseline_results, methods, gan_type):
    """Compare GAN-augmented results vs baseline with statistical significance tests"""
    print(f"\n{'='*70}")
    print(f"GAN vs BASELINE STATISTICAL COMPARISON ({gan_type})")
    print(f"{'='*70}")
    
    comparison_results = {}
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    print(f"\nPAIRED T-TESTS: {gan_type} vs Baseline")
    print("(p < 0.05 indicates statistically significant improvement)")
    print("-" * 90)
    print(f"{'Method':<18} | {'Metric':<10} | {'Baseline':<12} | {'GAN-Aug':<12} | {'Δ Mean':<10} | {'p-value':<8} | {'Sig':<5}")
    print("-" * 90)
    
    for method in methods:
        comparison_results[method] = {}
        
        for metric in metrics:
            # Extract values for paired t-test
            baseline_values = [fold_result[method][metric] for fold_result in baseline_results]
            gan_values = [fold_result[method][metric] for fold_result in gan_results]
            
            # Calculate means and improvement
            baseline_mean = np.mean(baseline_values)
            gan_mean = np.mean(gan_values)
            improvement = gan_mean - baseline_mean
            
            # Perform paired t-test
            statistic, p_value = stats.ttest_rel(gan_values, baseline_values)
            
            # Determine significance
            if p_value < 0.001:
                significance = "***"
            elif p_value < 0.01:
                significance = "**"
            elif p_value < 0.05:
                significance = "*"
            else:
                significance = "ns"
            
            # Store results
            comparison_results[method][metric] = {
                'baseline_mean': baseline_mean,
                'gan_mean': gan_mean,
                'improvement': improvement,
                'improvement_pct': (improvement / baseline_mean * 100) if baseline_mean > 0 else 0,
                'p_value': p_value,
                'significance': significance,
                'statistic': statistic
            }
            
            # Print results
            print(f"{method:<18} | {metric:<10} | {baseline_mean:<12.4f} | {gan_mean:<12.4f} | "
                  f"{improvement:<+10.4f} | {p_value:<8.4f} | {significance:<5}")
    
    # Summary of improvements
    print(f"\n{'='*50}")
    print("IMPROVEMENT SUMMARY")
    print(f"{'='*50}")
    
    significant_improvements = 0
    total_comparisons = len(methods) * len(metrics)
    
    for method in methods:
        method_improvements = []
        for metric in metrics:
            result = comparison_results[method][metric]
            if result['p_value'] < 0.05 and result['improvement'] > 0:
                significant_improvements += 1
                method_improvements.append(f"{metric}: +{result['improvement_pct']:.1f}%")
        
        if method_improvements:
            print(f"{method}: {', '.join(method_improvements)}")
        else:
            print(f"{method}: No significant improvements")
    
    improvement_rate = significant_improvements / total_comparisons * 100
    print(f"\nOverall: {significant_improvements}/{total_comparisons} comparisons show significant improvement ({improvement_rate:.1f}%)")
    
    return comparison_results

def rank_methods_with_std(aggregated_results, methods):
    """Rank methods based on multiple criteria with consideration for standard deviation"""
    print(f"\n{'='*50}")
    print("ENHANCED METHOD RANKING")
    print(f"{'='*50}")
    
    rankings = {}
    
    # Rank by F1 score (mean)
    f1_scores = [(method, aggregated_results[method]['f1']['mean'], aggregated_results[method]['f1']['std']) 
                 for method in methods]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['f1'] = f1_scores
    
    # Rank by accuracy (mean)
    accuracies = [(method, aggregated_results[method]['accuracy']['mean'], aggregated_results[method]['accuracy']['std']) 
                  for method in methods]
    accuracies.sort(key=lambda x: x[1], reverse=True)
    rankings['accuracy'] = accuracies
    
    # Rank by stability (lowest coefficient of variation)
    stability_scores = []
    for method in methods:
        f1_mean = aggregated_results[method]['f1']['mean']
        f1_std = aggregated_results[method]['f1']['std']
        cv = f1_std / f1_mean if f1_mean > 0 else float('inf')
        stability_scores.append((method, cv, f1_mean))
    stability_scores.sort(key=lambda x: x[1])
    rankings['stability'] = stability_scores
    
    # Rank by balanced score (average of precision and recall)
    balanced_scores = []
    for method in methods:
        prec = aggregated_results[method]['precision']['mean']
        rec = aggregated_results[method]['recall']['mean']
        balanced = (prec + rec) / 2
        balanced_std = np.sqrt((aggregated_results[method]['precision']['std']**2 + 
                               aggregated_results[method]['recall']['std']**2) / 2)
        balanced_scores.append((method, balanced, balanced_std))
    balanced_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['balanced'] = balanced_scores
    
    # Print rankings with standard deviations
    print("\nRANKING BY F1 SCORE (Mean ± Std):")
    for i, (method, score, std) in enumerate(f1_scores, 1):
        print(f"  {i}. {method:<22s}: {score:.4f} ± {std:.4f}")
    
    print("\nRANKING BY ACCURACY (Mean ± Std):")
    for i, (method, score, std) in enumerate(accuracies, 1):
        print(f"  {i}. {method:<22s}: {score:.4f} ± {std:.4f}")
    
    print("\nRANKING BY STABILITY (Coefficient of Variation):")
    for i, (method, cv, f1_mean) in enumerate(stability_scores, 1):
        print(f"  {i}. {method:<22s}: CV = {cv:.4f} (F1 = {f1_mean:.4f})")
    
    print("\nRANKING BY BALANCED SCORE (Precision + Recall)/2:")
    for i, (method, score, std) in enumerate(balanced_scores, 1):
        print(f"  {i}. {method:<22s}: {score:.4f} ± {std:.4f}")
    
    return rankings

def visualize_results_with_comparison(gan_aggregated, baseline_aggregated, gan_fold_results, 
                                    baseline_fold_results, methods, gan_type):
    """Create comprehensive visualizations comparing GAN vs baseline results"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 3, figsize=(24, 18))
    fig.suptitle(f'Comprehensive Anomaly Detection Results: {gan_type} vs Baseline', 
                 fontsize=16, fontweight='bold')
    
    # Plot 1: Mean performance comparison (GAN vs Baseline)
    ax1 = axes[0, 0]
    x = np.arange(len(methods))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        baseline_means = [baseline_aggregated[method][metric]['mean'] for method in methods]
        gan_means = [gan_aggregated[method][metric]['mean'] for method in methods]
        baseline_stds = [baseline_aggregated[method][metric]['std'] for method in methods]
        gan_stds = [gan_aggregated[method][metric]['std'] for method in methods]
        
        offset = (i - 1.5) * width / len(metrics)
        ax1.errorbar(x + offset - width/4, baseline_means, yerr=baseline_stds, 
                    fmt='o-', label=f'{metric} (Baseline)', alpha=0.7, capsize=3)
        ax1.errorbar(x + offset + width/4, gan_means, yerr=gan_stds, 
                    fmt='s-', label=f'{metric} ({gan_type})', alpha=0.7, capsize=3)
    
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Score')
    ax1.set_title('Performance Comparison: GAN vs Baseline')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Improvement heatmap
    ax2 = axes[0, 1]
    improvement_matrix = np.zeros((len(methods), len(metrics)))
    
    for i, method in enumerate(methods):
        for j, metric in enumerate(metrics):
            baseline_mean = baseline_aggregated[method][metric]['mean']
            gan_mean = gan_aggregated[method][metric]['mean']
            improvement = ((gan_mean - baseline_mean) / baseline_mean * 100) if baseline_mean > 0 else 0
            improvement_matrix[i, j] = improvement
    
    im = ax2.imshow(improvement_matrix, cmap='RdYlGn', aspect='auto', vmin=-10, vmax=10)
    ax2.set_xticks(range(len(metrics)))
    ax2.set_yticks(range(len(methods)))
    ax2.set_xticklabels(metrics)
    ax2.set_yticklabels(methods)
    ax2.set_title(f'Improvement Heatmap (%)\n{gan_type} vs Baseline')
    
    # Add text annotations
    for i in range(len(methods)):
        for j in range(len(metrics)):
            text = ax2.text(j, i, f'{improvement_matrix[i, j]:.1f}%', 
                           ha="center", va="center", color="black", fontweight='bold')
    
    plt.colorbar(im, ax=ax2)
    
    # Plot 3: F1 Score distribution comparison
    ax3 = axes[0, 2]
    baseline_f1_data = []
    gan_f1_data = []
    
    for method in methods:
        baseline_f1_values = [fold_result[method]['f1'] for fold_result in baseline_fold_results]
        gan_f1_values = [fold_result[method]['f1'] for fold_result in gan_fold_results]
        baseline_f1_data.append(baseline_f1_values)
        gan_f1_data.append(gan_f1_values)
    
    positions1 = np.arange(1, len(methods) * 2, 2)
    positions2 = np.arange(2, len(methods) * 2 + 1, 2)
    
    bp1 = ax3.boxplot(baseline_f1_data, positions=positions1, widths=0.6, 
                      patch_artist=True, boxprops=dict(facecolor='lightblue'))
    bp2 = ax3.boxplot(gan_f1_data, positions=positions2, widths=0.6, 
                      patch_artist=True, boxprops=dict(facecolor='lightgreen'))
    
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score Distribution Across Folds')
    ax3.set_xticks(np.arange(1.5, len(methods) * 2, 2))
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.legend([bp1["boxes"][0], bp2["boxes"][0]], ['Baseline', gan_type])
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Standard deviation comparison
    ax4 = axes[1, 0]
    metric_stds_baseline = []
    metric_stds_gan = []
    
    for metric in metrics:
        baseline_stds = [baseline_aggregated[method][metric]['std'] for method in methods]
        gan_stds = [gan_aggregated[method][metric]['std'] for method in methods]
        metric_stds_baseline.append(baseline_stds)
        metric_stds_gan.append(gan_stds)
    
    x = np.arange(len(methods))
    width = 0.35
    
    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width / len(metrics)
        ax4.bar(x + offset - width/4, metric_stds_baseline[i], width/len(metrics), 
               label=f'{metric} (Baseline)', alpha=0.7)
        ax4.bar(x + offset + width/4, metric_stds_gan[i], width/len(metrics), 
               label=f'{metric} ({gan_type})', alpha=0.7)
    
    ax4.set_xlabel('Methods')
    ax4.set_ylabel('Standard Deviation')
    ax4.set_title('Performance Stability Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Continue with remaining plots...
    # Plot 5: Statistical significance visualization
    ax5 = axes[1, 1]
    # Create significance matrix
    sig_matrix = np.zeros((len(methods), len(metrics)))
    
    # This would need the comparison results to show p-values
    # For now, create a placeholder
    ax5.text(0.5, 0.5, 'Statistical Significance\nMatrix\n(p-values)', 
            ha='center', va='center', transform=ax5.transAxes, fontsize=12)
    ax5.set_title('Statistical Significance')
    
    # Plot 6: Coefficient of variation
    ax6 = axes[1, 2]
    cv_baseline = []
    cv_gan = []
    
    for method in methods:
        for metric in metrics:
            baseline_mean = baseline_aggregated[method][metric]['mean']
            baseline_std = baseline_aggregated[method][metric]['std']
            gan_mean = gan_aggregated[method][metric]['mean']
            gan_std = gan_aggregated[method][metric]['std']
            
            cv_b = baseline_std / baseline_mean if baseline_mean > 0 else 0
            cv_g = gan_std / gan_mean if gan_mean > 0 else 0
            cv_baseline.append(cv_b)
            cv_gan.append(cv_g)
    
    ax6.scatter(cv_baseline, cv_gan, alpha=0.7, s=50)
    ax6.plot([0, max(cv_baseline + cv_gan)], [0, max(cv_baseline + cv_gan)], 'r--', alpha=0.5)
    ax6.set_xlabel('Baseline CV')
    ax6.set_ylabel(f'{gan_type} CV')
    ax6.set_title('Coefficient of Variation Comparison\n(Lower is more stable)')
    ax6.grid(True, alpha=0.3)
    
    # Plot 7-9: Individual metric comparisons
    for idx, metric in enumerate(['accuracy', 'precision', 'recall']):
        ax = axes[2, idx]
        baseline_values = [baseline_aggregated[method][metric]['mean'] for method in methods]
        gan_values = [gan_aggregated[method][metric]['mean'] for method in methods]
        baseline_stds = [baseline_aggregated[method][metric]['std'] for method in methods]
        gan_stds = [gan_aggregated[method][metric]['std'] for method in methods]
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax.bar(x - width/2, baseline_values, width, yerr=baseline_stds, 
               label='Baseline', alpha=0.7, capsize=5)
        ax.bar(x + width/2, gan_values, width, yerr=gan_stds, 
               label=gan_type, alpha=0.7, capsize=5)
        
        ax.set_xlabel('Methods')
        ax.set_ylabel(metric.capitalize())
        ax.set_title(f'{metric.capitalize()} Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def provide_enhanced_recommendations(gan_aggregated, baseline_aggregated, rankings, methods, 
                                   comparison_results, gan_type):
    """Provide enhanced recommendations based on GAN vs baseline analysis"""
    print(f"\n{'='*70}")
    print(f"ENHANCED ANOMALY DETECTION RECOMMENDATIONS ({gan_type})")
    print(f"{'='*70}")
    
    # Best overall method from GAN results
    best_f1_method = rankings['f1'][0][0]
    best_f1_score = rankings['f1'][0][1]
    best_f1_std = rankings['f1'][0][2]
    
    most_stable_method = rankings['stability'][0][0]
    stability_cv = rankings['stability'][0][1]
    
    print(f"\n🏆 BEST METHODS WITH {gan_type}:")
    print(f"   • Best F1 Score: {best_f1_method} ({best_f1_score:.4f} ± {best_f1_std:.4f})")
    print(f"   • Most Stable: {most_stable_method} (CV = {stability_cv:.4f})")
    
    # GAN effectiveness analysis
    print(f"\n📈 {gan_type} EFFECTIVENESS:")
    total_improvements = 0
    significant_improvements = 0
    
    for method in methods:
        method_improvements = 0
        method_significant = 0
        
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            result = comparison_results[method][metric]
            if result['improvement'] > 0:
                method_improvements += 1
            if result['p_value'] < 0.05 and result['improvement'] > 0:
                method_significant += 1
                significant_improvements += 1
            total_improvements += 1
        
        improvement_rate = method_improvements / 4 * 100
        significance_rate = method_significant / 4 * 100
        
        print(f"   • {method}: {improvement_rate:.0f}% metrics improved, "
              f"{significance_rate:.0f}% significantly")
    
    overall_significance_rate = significant_improvements / (len(methods) * 4) * 100
    
    if overall_significance_rate > 50:
        gan_effectiveness = "Highly Effective"
    elif overall_significance_rate > 25:
        gan_effectiveness = "Moderately Effective"
    elif overall_significance_rate > 10:
        gan_effectiveness = "Slightly Effective"
    else:
        gan_effectiveness = "Not Effective"
    
    print(f"\n🎯 {gan_type} OVERALL EFFECTIVENESS: {gan_effectiveness}")
    print(f"   • {significant_improvements}/{len(methods) * 4} comparisons show significant improvement")
    print(f"   • Success Rate: {overall_significance_rate:.1f}%")
    
    # Method-specific recommendations
    print(f"\n📊 METHOD-SPECIFIC INSIGHTS:")
    for method in methods:
        gan_f1 = gan_aggregated[method]['f1']['mean']
        baseline_f1 = baseline_aggregated[method]['f1']['mean']
        gan_std = gan_aggregated[method]['f1']['std']
        baseline_std = baseline_aggregated[method]['f1']['std']
        
        improvement = gan_f1 - baseline_f1
        stability_improvement = baseline_std - gan_std
        
        characteristics = []
        if improvement > 0.01:
            characteristics.append("Performance Boost")
        if stability_improvement > 0.005:
            characteristics.append("Stability Gain")
        if gan_f1 > 0.8:
            characteristics.append("High Performance")
        if gan_std < 0.05:
            characteristics.append("Very Stable")
        
        if not characteristics:
            characteristics.append("No Significant Benefit")
        
        print(f"   • {method:<22s}: {', '.join(characteristics)}")
    
    # Final recommendations
    print(f"\n🎯 FINAL RECOMMENDATIONS:")
    
    if overall_significance_rate > 25:
        print(f"   ✅ {gan_type} data augmentation is RECOMMENDED")
        print(f"   • Use {best_f1_method} for best performance")
        print(f"   • Expected improvement: Significant in {overall_significance_rate:.0f}% of cases")
    else:
        print(f"   ❌ {gan_type} data augmentation shows LIMITED benefit")
        print(f"   • Consider baseline methods or alternative GAN architectures")
        print(f"   • Current success rate: {overall_significance_rate:.1f}%")
    
    print(f"\n💡 DEPLOYMENT STRATEGY:")
    if gan_effectiveness in ["Highly Effective", "Moderately Effective"]:
        print(f"   • Deploy {gan_type}-augmented {best_f1_method} in production")
        print(f"   • Monitor performance stability (current CV: {rankings['stability'][0][1]:.4f})")
        print(f"   • Set up A/B testing vs baseline for continuous validation")
    else:
        print(f"   • Stick with baseline methods for now")
        print(f"   • Investigate alternative data augmentation strategies")
        print(f"   • Consider ensemble methods combining multiple approaches")
    
    print(f"\n{'='*70}")

def aggregate_results(fold_results, methods):
    """Legacy function - redirect to enhanced version"""
    return aggregate_results_with_std(fold_results, methods, "Legacy")

def rank_methods(aggregated_results, methods):
    """Legacy function - redirect to enhanced version"""
    return rank_methods_with_std(aggregated_results, methods)

def visualize_results(aggregated_results, fold_results, methods):
    """Legacy function - create basic visualization"""
    print("Using basic visualization - consider using enhanced version for better insights")
    
    # Create basic visualization
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # F1 scores
    f1_means = [aggregated_results[method]['f1']['mean'] for method in methods]
    f1_stds = [aggregated_results[method]['f1']['std'] for method in methods]
    
    axes[0].bar(methods, f1_means, yerr=f1_stds, capsize=5, alpha=0.7)
    axes[0].set_ylabel('F1 Score')
    axes[0].set_title('F1 Score Comparison')
    axes[0].set_xticklabels(methods, rotation=45, ha='right')
    axes[0].grid(True, alpha=0.3)
    
    # Box plots
    f1_data = []
    for method in methods:
        f1_values = [fold_result[method]['f1'] for fold_result in fold_results]
        f1_data.append(f1_values)
    
    axes[1].boxplot(f1_data, labels=methods)
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score Distribution')
    axes[1].set_xticklabels(methods, rotation=45, ha='right')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def provide_recommendations(aggregated_results, rankings, methods):
    """Legacy function - redirect to basic recommendations"""
    print("Using basic recommendations - consider using enhanced version for better insights")
    
    best_f1_method = rankings['f1'][0][0]
    best_f1_score = rankings['f1'][0][1]
    
    print(f"\n🏆 BEST METHOD: {best_f1_method} (F1: {best_f1_score:.4f})")
    print(f"\n📊 ALL METHODS RANKED BY F1:")
    for i, (method, score, std) in enumerate(rankings['f1'], 1):
        print(f"   {i}. {method}: {score:.4f} ± {std:.4f}")

def rank_methods(aggregated_results, methods):
    """Rank methods based on multiple criteria"""
    print(f"\n{'='*50}")
    print("METHOD RANKING")
    print(f"{'='*50}")
    
    rankings = {}
    
    # Rank by F1 score
    f1_scores = [(method, aggregated_results[method]['f1']['mean']) for method in methods]
    f1_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['f1'] = f1_scores
    
    # Rank by accuracy
    accuracies = [(method, aggregated_results[method]['accuracy']['mean']) for method in methods]
    accuracies.sort(key=lambda x: x[1], reverse=True)
    rankings['accuracy'] = accuracies
    
    # Rank by balanced score (average of precision and recall)
    balanced_scores = []
    for method in methods:
        prec = aggregated_results[method]['precision']['mean']
        rec = aggregated_results[method]['recall']['mean']
        balanced = (prec + rec) / 2
        balanced_scores.append((method, balanced))
    balanced_scores.sort(key=lambda x: x[1], reverse=True)
    rankings['balanced'] = balanced_scores
    
    # Print rankings
    print("\nRANKING BY F1 SCORE:")
    for i, (method, score) in enumerate(f1_scores, 1):
        print(f"  {i}. {method:<22s}: {score:.4f}")
    
    print("\nRANKING BY ACCURACY:")
    for i, (method, score) in enumerate(accuracies, 1):
        print(f"  {i}. {method:<22s}: {score:.4f}")
    
    print("\nRANKING BY BALANCED SCORE (Precision + Recall)/2:")
    for i, (method, score) in enumerate(balanced_scores, 1):
        print(f"  {i}. {method:<22s}: {score:.4f}")
    
    return rankings

def visualize_results(aggregated_results, fold_results, methods):
    """Create comprehensive visualizations of the results"""
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Anomaly Detection Results', fontsize=16, fontweight='bold')
    
    # Plot 1: Mean performance comparison
    ax1 = axes[0, 0]
    metric_means = []
    for metric in metrics:
        means = [aggregated_results[method][metric]['mean'] for method in methods]
        metric_means.append(means)
    
    x = np.arange(len(methods))
    width = 0.2
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax1.bar(x + i * width, metric_means[i], width, label=metric.capitalize(), color=color, alpha=0.7)
    
    ax1.set_xlabel('Methods')
    ax1.set_ylabel('Score')
    ax1.set_title('Mean Performance Comparison')
    ax1.set_xticks(x + width * 1.5)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: F1 Score with error bars
    ax2 = axes[0, 1]
    f1_means = [aggregated_results[method]['f1']['mean'] for method in methods]
    f1_stds = [aggregated_results[method]['f1']['std'] for method in methods]
    
    bars = ax2.bar(methods, f1_means, yerr=f1_stds, capsize=5, color='skyblue', alpha=0.7)
    ax2.set_ylabel('F1 Score')
    ax2.set_title('F1 Score Comparison (with std dev)')
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Box plots for F1 scores across folds
    ax3 = axes[0, 2]
    f1_data = []
    for method in methods:
        f1_values = [fold_result[method]['f1'] for fold_result in fold_results]
        f1_data.append(f1_values)
    
    bp = ax3.boxplot(f1_data, labels=methods, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightyellow', 'lightcoral', 'lightpink']
    for patch, color in zip(bp['boxes'], colors[:len(methods)]):
        patch.set_facecolor(color)
    
    ax3.set_ylabel('F1 Score')
    ax3.set_title('F1 Score Distribution Across Folds')
    ax3.set_xticklabels(methods, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Precision vs Recall scatter
    ax4 = axes[1, 0]
    for i, method in enumerate(methods):
        prec_mean = aggregated_results[method]['precision']['mean']
        rec_mean = aggregated_results[method]['recall']['mean']
        prec_std = aggregated_results[method]['precision']['std']
        rec_std = aggregated_results[method]['recall']['std']
        
        ax4.errorbar(rec_mean, prec_mean, xerr=rec_std, yerr=prec_std, 
                    fmt='o', markersize=8, label=method, capsize=5)
    
    ax4.set_xlabel('Recall')
    ax4.set_ylabel('Precision')
    ax4.set_title('Precision vs Recall (with std dev)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    
    # Plot 5: Performance consistency (coefficient of variation)
    ax5 = axes[1, 1]
    cv_data = []
    for metric in metrics:
        cvs = []
        for method in methods:
            mean_val = aggregated_results[method][metric]['mean']
            std_val = aggregated_results[method][metric]['std']
            cv = std_val / mean_val if mean_val > 0 else 0
            cvs.append(cv)
        cv_data.append(cvs)
    
    x = np.arange(len(methods))
    width = 0.2
    
    for i, (cv_values, metric, color) in enumerate(zip(cv_data, metrics, colors)):
        ax5.bar(x + i * width, cv_values, width, label=metric.capitalize(), color=color, alpha=0.7)
    
    ax5.set_xlabel('Methods')
    ax5.set_ylabel('Coefficient of Variation')
    ax5.set_title('Performance Consistency (Lower is Better)')
    ax5.set_xticks(x + width * 1.5)
    ax5.set_xticklabels(methods, rotation=45, ha='right')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Overall ranking
    ax6 = axes[1, 2]
    overall_ranks = []
    for method in methods:
        ranks = []
        for metric in metrics:
            # Get rank for this method in this metric
            metric_values = [(aggregated_results[m][metric]['mean'], i) for i, m in enumerate(methods)]
            metric_values.sort(reverse=True)
            rank = next(i for i, (_, idx) in enumerate(metric_values) if idx == methods.index(method)) + 1
            ranks.append(rank)
        overall_ranks.append(np.mean(ranks))
    
    # Sort methods by overall rank
    method_rank_pairs = list(zip(methods, overall_ranks))
    method_rank_pairs.sort(key=lambda x: x[1])
    
    ranked_methods = [pair[0] for pair in method_rank_pairs]
    ranked_scores = [pair[1] for pair in method_rank_pairs]
    
    bars = ax6.barh(range(len(ranked_methods)), ranked_scores, color='gold', alpha=0.7)
    ax6.set_yticks(range(len(ranked_methods)))
    ax6.set_yticklabels(ranked_methods)
    ax6.set_xlabel('Average Rank (Lower is Better)')
    ax6.set_title('Overall Method Ranking')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def provide_recommendations(aggregated_results, rankings, methods):
    """Provide recommendations based on the analysis"""
    print(f"\n{'='*70}")
    print("ANOMALY DETECTION RECOMMENDATIONS")
    print(f"{'='*70}")
    
    # Best overall method
    best_f1_method = rankings['f1'][0][0]
    best_f1_score = rankings['f1'][0][1]
    
    best_acc_method = rankings['accuracy'][0][0]
    best_acc_score = rankings['accuracy'][0][1]
    
    print(f"\n🏆 BEST METHODS:")
    print(f"   • Best F1 Score: {best_f1_method} ({best_f1_score:.4f})")
    print(f"   • Best Accuracy: {best_acc_method} ({best_acc_score:.4f})")
    
    # Method characteristics
    print(f"\n📊 METHOD CHARACTERISTICS:")
    for method in methods:
        prec = aggregated_results[method]['precision']['mean']
        rec = aggregated_results[method]['recall']['mean']
        prec_std = aggregated_results[method]['precision']['std']
        rec_std = aggregated_results[method]['recall']['std']
        
        if prec > rec + 0.05:
            characteristic = "High Precision (fewer false alarms)"
        elif rec > prec + 0.05:
            characteristic = "High Recall (catches more anomalies)"
        else:
            characteristic = "Balanced precision and recall"
            
        stability = "Stable" if prec_std < 0.1 and rec_std < 0.1 else "Variable"
        
        print(f"   • {method:<22s}: {characteristic}, {stability}")
    
    # Use case recommendations
    print(f"\n🎯 USE CASE RECOMMENDATIONS:")
    print(f"   • For Critical Systems (minimize false negatives): Use method with highest recall")
    print(f"   • For Cost-Sensitive Systems (minimize false alarms): Use method with highest precision")
    print(f"   • For Balanced Performance: Use {best_f1_method}")
    print(f"   • For Simplicity: Use 95th_Percentile - no hyperparameter tuning needed")
    print(f"   • For Robustness: Use OneClass_SVM - adapts to data distribution")
    
    print(f"\n{'='*70}")
