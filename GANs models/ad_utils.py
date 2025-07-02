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

def train_ocsvm_on_features(features, contamination=0.1):
    """
    Train One-Class SVM on autoencoder features (encoded representations)
    features: encoded features from autoencoder encoder
    """
    from sklearn.svm import OneClassSVM
    from sklearn.preprocessing import StandardScaler
    
    # Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # Train One-Class SVM
    ocsvm = OneClassSVM(kernel='rbf', gamma='scale')
    ocsvm.fit(features_scaled)
    
    return ocsvm, scaler

def predict_with_ocsvm(model, ocsvm, scaler, X_data):
    """
    Get OCSVM predictions using autoencoder encoded features
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
    predictions = ocsvm.predict(encoded_scaled)
    
    # Convert to anomaly labels (1 = anomaly, 0 = normal)
    anomaly_preds = (predictions == -1).astype(int)
    
    return anomaly_preds


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
                                                  n_splits=5, epochs=20, batch_size=128):
    """
    Unified comprehensive cross-validation experiment for anomaly detection
    
    Args:
        normal_data: Normal samples for training/testing
        faulty_data: Faulty samples for training/testing
        generated_data: Optional generated samples for data augmentation
        n_splits: Number of cross-validation folds
        epochs: Training epochs for autoencoder
        batch_size: Batch size for training

    Returns:
        aggregated_results: Averaged metrics across folds
        fold_results: Individual fold results
        rankings: Method rankings by different criteria
    """
    print(f"\n{'='*70}")
    print("COMPREHENSIVE ANOMALY DETECTION CROSS-VALIDATION EXPERIMENT")
    print(f"{'='*70}")
    print(f"Normal samples: {len(normal_data)}")
    print(f"Faulty samples: {len(faulty_data)}")
    if generated_data is not None:
        print(f"Generated samples: {len(generated_data)}")
    print(f"Cross-validation folds: {n_splits}")
    
    # Combine all data for stratified splitting
    all_data = np.concatenate([normal_data, faulty_data], axis=0)
    normal_labels = np.zeros(len(normal_data))
    faulty_labels = np.ones(len(faulty_data))
    all_labels = np.concatenate([normal_labels, faulty_labels], axis=0)
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Initialize results storage
    methods = ['F1_Threshold', 'Accuracy_Threshold', '95th_Percentile', '90th_Percentile', 'OneClass_SVM']
    fold_results = []
    
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
        train_normal_fold = train_data_fold[normal_train_mask]
        
        print(f"Fold {fold+1} - Train normal: {len(train_normal_fold)}")
        print(f"Fold {fold+1} - Test: {len(test_data_fold)} ({np.sum(test_labels_fold==0)} normal, {np.sum(test_labels_fold==1)} faulty)")
        
        # Optionally augment normal training data with generated samples
        if generated_data is not None:
            augmented_normal_data = np.concatenate([generated_data, train_normal_fold], axis=0)
            print(f"Fold {fold+1} - Augmented normal data: {len(augmented_normal_data)} samples")
        else:
            augmented_normal_data = train_normal_fold
            print(f"Fold {fold+1} - Using original normal data only")
        
        # Process data through feature extraction pipeline
        print("Processing data through feature extraction...")
        augmented_normal_features = process_dataset_multichannel(augmented_normal_data, device)
        test_features = process_dataset_multichannel(test_data_fold, device)
        
        # Add channel dimension for autoencoder compatibility
        augmented_normal_features = augmented_normal_features[:, np.newaxis, :]
        test_features = test_features[:, np.newaxis, :]
        
        # Train autoencoder on augmented normal data
        print("Training autoencoder...")
        model = train_autoencoder(augmented_normal_features.reshape(-1, 4096), device, epochs, batch_size)
        
        # Compute reconstruction errors on test set
        test_errors = compute_reconstruction_loss(model, test_features)
        
        # Evaluate all methods
        fold_result = {}
        
        # Method 1: F1-optimized threshold
        f1_threshold, _ = find_best_threshold(test_errors, test_labels_fold)
        f1_preds = (test_errors > f1_threshold).astype(int)
        fold_result['F1_Threshold'] = {
            'accuracy': accuracy_score(test_labels_fold, f1_preds),
            'precision': precision_score(test_labels_fold, f1_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, f1_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, f1_preds, zero_division=0),
            'threshold': f1_threshold
        }
        
        # Method 2: Accuracy-optimized threshold
        acc_threshold, _ = find_best_threshold_using_accuracy(test_errors, test_labels_fold)
        acc_preds = (test_errors > acc_threshold).astype(int)
        fold_result['Accuracy_Threshold'] = {
            'accuracy': accuracy_score(test_labels_fold, acc_preds),
            'precision': precision_score(test_labels_fold, acc_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, acc_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, acc_preds, zero_division=0),
            'threshold': acc_threshold
        }
        
        # Method 3: 95th percentile threshold
        p95_threshold = get_percentile_threshold(test_errors, 95)
        p95_preds = (test_errors > p95_threshold).astype(int)
        fold_result['95th_Percentile'] = {
            'accuracy': accuracy_score(test_labels_fold, p95_preds),
            'precision': precision_score(test_labels_fold, p95_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, p95_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, p95_preds, zero_division=0),
            'threshold': p95_threshold
        }
        
        # Method 4: 90th percentile threshold
        p90_threshold = get_percentile_threshold(test_errors, 90)
        p90_preds = (test_errors > p90_threshold).astype(int)
        fold_result['90th_Percentile'] = {
            'accuracy': accuracy_score(test_labels_fold, p90_preds),
            'precision': precision_score(test_labels_fold, p90_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, p90_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, p90_preds, zero_division=0),
            'threshold': p90_threshold
        }
        
        # Method 5: One-Class SVM
        print("Training One-Class SVM...")
        # Get encoded features for normal data
        model.eval()
        with torch.no_grad():
            normal_features_tensor = torch.tensor(augmented_normal_features.reshape(-1, 4096), dtype=torch.float32).to(device)
            normal_encoded = model.encoder(normal_features_tensor).cpu().numpy()
            normal_encoded = normal_encoded.reshape(len(augmented_normal_features), -1, normal_encoded.shape[-1]).mean(axis=1)
        
        ocsvm, scaler = train_ocsvm_on_features(normal_encoded)
        ocsvm_preds = predict_with_ocsvm(model, ocsvm, scaler, test_features)
        fold_result['OneClass_SVM'] = {
            'accuracy': accuracy_score(test_labels_fold, ocsvm_preds),
            'precision': precision_score(test_labels_fold, ocsvm_preds, zero_division=0),
            'recall': recall_score(test_labels_fold, ocsvm_preds, zero_division=0),
            'f1': f1_score(test_labels_fold, ocsvm_preds, zero_division=0),
            'threshold': 'OCSVM'
        }
        
        # Print fold results
        print(f"\nFold {fold+1} Results:")
        print("-" * 50)
        for method in methods:
            result = fold_result[method]
            print(f"{method:18} | Acc: {result['accuracy']:.4f} | Prec: {result['precision']:.4f} | Rec: {result['recall']:.4f} | F1: {result['f1']:.4f}")
        
        fold_results.append(fold_result)
    
    # Aggregate results across folds
    aggregated_results = aggregate_results(fold_results, methods)
    
    # Rank methods
    rankings = rank_methods(aggregated_results, methods)
    
    # Create visualizations
    visualize_results(aggregated_results, fold_results, methods)
    
    # Provide recommendations
    provide_recommendations(aggregated_results, rankings, methods)
    
    return aggregated_results, fold_results, rankings

def aggregate_results(fold_results, methods):
    """Aggregate results across all folds and compute statistics"""
    print(f"\n{'='*70}")
    print("RESULTS AGGREGATION & STATISTICAL ANALYSIS")
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
                'values': values
            }
    
    # Print detailed results
    print("\nDETAILED RESULTS SUMMARY:")
    print("-" * 80)
    print(f"{'Method':<18} | {'Accuracy':<8} | {'Precision':<9} | {'Recall':<8} | {'F1-Score':<8}")
    print("-" * 80)
    
    for method in methods:
        acc_mean = aggregated_results[method]['accuracy']['mean']
        prec_mean = aggregated_results[method]['precision']['mean']
        rec_mean = aggregated_results[method]['recall']['mean']
        f1_mean = aggregated_results[method]['f1']['mean']
        
        print(f"{method:<18} | {acc_mean:<8.4f} | {prec_mean:<9.4f} | {rec_mean:<8.4f} | {f1_mean:<8.4f}")
    
    # Statistical significance testing
    print(f"\n{'='*50}")
    print("STATISTICAL SIGNIFICANCE TESTING")
    print(f"{'='*50}")
    
    # Perform pairwise t-tests for F1 scores
    f1_data = {method: aggregated_results[method]['f1']['values'] for method in methods}
    
    print("\nPairwise t-tests for F1 scores:")
    print("(p < 0.05 indicates statistically significant difference)")
    print("-" * 70)
    
    method_pairs = [(i, j) for i in range(len(methods)) for j in range(i+1, len(methods))]
    
    for i, j in method_pairs:
        method1, method2 = methods[i], methods[j]
        values1 = f1_data[method1]
        values2 = f1_data[method2]
        
        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(values1, values2)
        significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
        
        print(f"{method1:18s} vs {method2:18s}: t={statistic:6.3f}, p={p_value:.4f} {significance}")
    
    return aggregated_results

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
