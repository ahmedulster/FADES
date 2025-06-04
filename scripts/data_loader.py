import os
import pickle
import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATASET_PATH = "/Users/ayesha/Desktop/Dessertation_Ahmed/Federated_learning_wesad/dataset/WESAD"
VALID_SUBJECTS = range(2, 17)  # S2-S17 for fewer subjects
REQUIRED_SIGNAL_KEYS = ['wrist', 'chest']
SEQUENCE_LENGTH = 60
SEQUENCE_STRIDE = 30

# CRITICAL FIX: Define fixed label map based on WESAD dataset labels
# WESAD labels: 0 (undefined/transient), 1 (baseline), 2 (stress), 3 (amusement), 4 (meditation)
FIXED_LABEL_MAP = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4}

class DataValidationError(Exception):
    """Custom exception for data validation failures"""
    pass

class DataLoadingError(Exception):
    """Custom exception for data loading failures"""
    pass

def verify_dataset_structure(data: Dict[str, Any], subject: str) -> None:
    """Simple structure validation with more flexible sensor checks"""
    errors = []

    # Check top-level structure
    for key in ['signal', 'label']:
        if key not in data:
            errors.append(f"Missing required top-level key: {key}")

    if errors:
        raise DataValidationError(f"{subject}: Initial validation failed - {', '.join(errors)}")

    # Validate signal structure
    try:
        signals = data['signal']
        if not isinstance(signals, dict):
            raise DataValidationError(f"{subject}: Signal data is not a dictionary")

        available_signals = list(signals.keys())
        logger.info(f"{subject}: Available signal sources: {available_signals}")
        
        found_signals = []
        for req_key in REQUIRED_SIGNAL_KEYS:
            if req_key in signals:
                found_signals.append(req_key)
                
        if not found_signals:
            errors.append(f"None of the required signal keys found: {REQUIRED_SIGNAL_KEYS}")
        else:
            logger.info(f"{subject}: Found usable signal sources: {found_signals}")

        labels = data['label']
        if not isinstance(labels, np.ndarray) or labels.ndim != 1:
            errors.append("Labels must be 1D numpy array")
            
        unique_labels = np.unique(labels)
        logger.info(f"{subject}: Unique labels in dataset: {unique_labels}")

    except KeyError as e:
        errors.append(f"Missing key in signal data: {str(e)}")
    except Exception as e:
        errors.append(f"Error during validation: {str(e)}")

    if errors:
        error_msg = f"{subject}: Data validation failed - {', '.join(errors)}"
        logger.error(error_msg)
        raise DataValidationError(error_msg)

def load_subject(subject_id: str) -> Dict[str, Any]:
    """Load a subject's data with enhanced signal handling"""
    subject_folder = f"S{subject_id}"
    file_path = os.path.join(DATASET_PATH, subject_folder, f"{subject_folder}.pkl")
    
    if not os.path.exists(file_path):
        raise DataLoadingError(f"Subject data not found: {file_path}")

    encodings = ['latin1', 'bytes', 'utf-8', 'iso-8859-1']
    
    for encoding in encodings:
        try:
            with open(file_path, 'rb') as f:
                data = pickle.load(f, encoding=encoding)
                verify_dataset_structure(data, subject_folder)
                
                signals = data['signal']
                result = {'label': data['label']}
                
                unique_labels, counts = np.unique(data['label'], return_counts=True)
                logger.info(f"{subject_folder} full label distribution: {dict(zip(unique_labels, counts))}")
                
                if 'wrist' in signals and 'ACC' in signals['wrist']:
                    acc_data = signals['wrist']['ACC']
                    if acc_data.shape[0] != data['label'].shape[0]:
                        logger.warning(f"{subject_folder}: Wrist data/label length mismatch: {acc_data.shape[0]} vs {data['label'].shape[0]}")
                        
                        # Find common length and align data properly
                        min_length = min(acc_data.shape[0], data['label'].shape[0])
                        
                        # If too different, try to find proper alignment using signal characteristics
                        if abs(acc_data.shape[0] - data['label'].shape[0]) > min_length * 0.1:
                            logger.warning("Significant length difference, attempting to find proper alignment")
                            
                            # Use label transitions as alignment points
                            label_transitions = np.where(np.diff(data['label']) != 0)[0]
                            
                            # Use activity spikes in acceleration data
                            acc_magnitude = np.sqrt(np.sum(np.square(acc_data), axis=1))
                            acc_transitions = np.where(np.diff(acc_magnitude) > np.std(acc_magnitude))[0]
                            
                            # If we found potential alignment points, use them
                            if len(label_transitions) > 0 and len(acc_transitions) > 0:
                                # Find best alignment by trying different offsets
                                best_offset = 0
                                best_match = 0
                                
                                for offset in range(-100, 100):
                                    matches = 0
                                    for lt in label_transitions:
                                        for at in acc_transitions:
                                            if lt < data['label'].shape[0] and at < acc_data.shape[0]:
                                                if abs(lt - (at + offset)) < 10:  # Within 10 samples
                                                    matches += 1
                                    
                                    if matches > best_match:
                                        best_match = matches
                                        best_offset = offset
                                
                                logger.info(f"Found potential alignment with offset {best_offset}")
                                
                                # Apply offset if it seems reasonable
                                if best_offset != 0 and best_match > len(label_transitions) * 0.2:
                                    if best_offset > 0:
                                        if best_offset < acc_data.shape[0]:
                                            acc_data = acc_data[best_offset:]
                                    else:
                                        abs_offset = abs(best_offset)
                                        if abs_offset < data['label'].shape[0]:
                                            data['label'] = data['label'][abs_offset:]
                        
                        # Final trimming to common length
                        min_length = min(acc_data.shape[0], data['label'].shape[0])
                        acc_data = acc_data[:min_length]
                        result['label'] = data['label'][:min_length]
                    
                    result['acc'] = acc_data
                    result['source'] = 'wrist'
                    logger.info(f"Loaded {subject_folder} wrist ACC data with {encoding} encoding")
                
                if 'chest' in signals and 'ACC' in signals['chest']:
                    chest_acc = signals['chest']['ACC']
                    min_length = min(chest_acc.shape[0], data['label'].shape[0])
                    result['chest_acc'] = chest_acc[:min_length]
                
                return result
                
        except Exception as e:
            logger.warning(f"Failed to load {subject_folder} with {encoding}: {str(e)}")
            continue

    raise DataLoadingError(f"All encoding attempts failed for {subject_folder}")

def balance_classes(
    X: np.ndarray, 
    y: np.ndarray, 
    max_per_class: int = 10000, 
    min_samples_per_class: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Balance classes ensuring minimum samples per class
    
    Args:
        X: Feature array
        y: Label array
        max_per_class: Maximum samples per class (to prevent memory issues)
        min_samples_per_class: Minimum samples per class to ensure representation
    """
    class_counts = Counter(y)
    logger.info(f"Original class distribution: {class_counts}")
    
    if len(class_counts) <= 1:
        logger.warning("Only one class found, cannot balance")
        return X, y
    
    # FIXED: Ensure minimum samples per class while keeping reasonable maximum
    target_counts = {}
    for class_label, count in class_counts.items():
        # Set at least min_samples_per_class but no more than max_per_class
        target_counts[class_label] = min(max(count, min_samples_per_class), max_per_class)
    
    balanced_X = []
    balanced_y = []
    
    for class_label, target_count in target_counts.items():
        indices = np.where(y == class_label)[0]
        if len(indices) == 0:
            continue
        
        # Decide whether to oversample or undersample
        replace = len(indices) < target_count  # True if we need to oversample
        selected_indices = np.random.choice(indices, target_count, replace=replace)
        
        balanced_X.append(X[selected_indices])
        balanced_y.append(np.full(target_count, class_label))
    
    # Combine and shuffle
    balanced_X = np.vstack(balanced_X)
    balanced_y = np.concatenate(balanced_y)
    
    shuffle_idx = np.random.permutation(len(balanced_y))
    balanced_X = balanced_X[shuffle_idx]
    balanced_y = balanced_y[shuffle_idx]
    
    logger.info(f"Balanced class distribution: {Counter(balanced_y)}")
    return balanced_X, balanced_y

def create_sequences(
    X: np.ndarray, 
    y: np.ndarray, 
    sequence_length: int = SEQUENCE_LENGTH, 
    stride: int = SEQUENCE_STRIDE,
    majority_threshold: float = 0.5
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences with majority label threshold
    
    Args:
        X: Feature array
        y: Label array
        sequence_length: Length of each sequence
        stride: Step size between sequences
        majority_threshold: Minimum ratio required for majority class
    """
    X_sequences = []
    y_sequences = []
    
    if len(X) <= sequence_length:
        logger.warning(f"Data length ({len(X)}) <= sequence length ({sequence_length})")
        return np.empty((0, sequence_length, X.shape[1])), np.empty(0)
    
    for i in range(0, len(X) - sequence_length, stride):
        sequence_X = X[i:i+sequence_length]
        sequence_y = y[i:i+sequence_length]
        
        if len(sequence_X) < sequence_length:
            continue
        
        # FIXED: Only include sequences with clear majority class
        unique, counts = np.unique(sequence_y, return_counts=True)
        if len(unique) == 0:
            continue
            
        majority_label = unique[np.argmax(counts)]
        majority_ratio = counts.max() / len(sequence_y)
        
        # Only include sequences where majority class is sufficiently dominant
        if majority_ratio >= majority_threshold:
            X_sequences.append(sequence_X)
            y_sequences.append(majority_label)
    
    if len(X_sequences) == 0:
        logger.warning("No valid sequences created after applying majority threshold")
        return np.empty((0, sequence_length, X.shape[1])), np.empty(0)
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    # Verify shapes match
    if len(X_sequences) != len(y_sequences):
        min_len = min(len(X_sequences), len(y_sequences))
        X_sequences = X_sequences[:min_len]
        y_sequences = y_sequences[:min_len]
    
    logger.info(f"Created {len(X_sequences)} sequences with threshold {majority_threshold}")
    return X_sequences, y_sequences

def preprocess_data(
    subject_data: Dict, 
    use_temporal_split: bool = True, 
    balance: bool = True, 
    use_chest: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Preprocess data with fixed label mapping and filtering
    """
    if use_chest and 'chest_acc' in subject_data:
        X = subject_data['chest_acc']
        logger.info("Using chest acceleration data")
    else:
        X = subject_data['acc']
        logger.info("Using wrist acceleration data")
        
    y = subject_data['label']
    
    # FIXED: Use consistent label mapping across all subjects
    # Filter out labels not in our fixed label map
    valid_mask = np.isin(y, list(FIXED_LABEL_MAP.keys()))
    y_filtered = y[valid_mask]
    X_filtered = X[valid_mask]
    
    if len(y_filtered) == 0:
        raise ValueError("No valid labels after filtering")
    
    # Map to fixed labels using vectorized approach
    mapped_y = np.vectorize(FIXED_LABEL_MAP.get)(y_filtered)
    logger.info(f"Mapped label distribution: {Counter(mapped_y)}")
    
    # Balance classes if requested and multiple classes exist
    if balance and len(np.unique(mapped_y)) > 1:
        X_balanced, y_balanced = balance_classes(
            X_filtered, mapped_y, 
            min_samples_per_class=100  # Ensure at least 100 samples per class
        )
    else:
        X_balanced, y_balanced = X_filtered, mapped_y
    
    # Feature-wise normalization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_balanced)
    
    # Create sequences with majority threshold
    X_sequences, y_sequences = create_sequences(
        X_scaled, y_balanced, 
        majority_threshold=0.6  # Require 60% of window to be same class
    )
    
    if len(X_sequences) == 0:
        raise ValueError("No sequences could be created after applying majority threshold")
    
    # Reshape for CNN: (samples, timesteps, features, 1)
    X_sequences = X_sequences.reshape(-1, SEQUENCE_LENGTH, X_scaled.shape[1], 1)
    
    # Split data - with careful handling of class distribution
    if use_temporal_split:
        # Try temporal split but verify both splits have data
        split_idx = int(len(X_sequences) * 0.8)
        X_train, X_test = X_sequences[:split_idx], X_sequences[split_idx:]
        y_train, y_test = y_sequences[:split_idx], y_sequences[split_idx:]
        
        # Check if both splits have all classes
        train_classes = set(np.unique(y_train))
        test_classes = set(np.unique(y_test))
        
        # If classes are missing in either split, fall back to stratified split
        if len(train_classes) < len(np.unique(y_sequences)) or len(test_classes) < len(np.unique(y_sequences)):
            logger.warning("Temporal split resulted in missing classes. Using stratified split instead.")
            use_temporal_split = False
    
    if not use_temporal_split:
        # Use stratified split to maintain class distribution
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, 
                test_size=0.2, 
                stratify=y_sequences if len(np.unique(y_sequences)) > 1 else None,
                random_state=42
            )
        except ValueError as e:
            logger.warning(f"Stratified split failed: {str(e)}. Using regular split.")
            X_train, X_test, y_train, y_test = train_test_split(
                X_sequences, y_sequences, 
                test_size=0.2,
                random_state=42
            )
    
    # Log final class distribution
    logger.info(f"Final train class distribution: {Counter(y_train)}")
    logger.info(f"Final test class distribution: {Counter(y_test)}")
    
    return X_train, X_test, y_train, y_test

def load_wesad_data() -> Dict[str, Dict[str, np.ndarray]]:
    """Load and validate WESAD dataset for all subjects"""
    subjects_data = {}
    
    for subject_id in VALID_SUBJECTS:
        if subject_id == 12:  # Skip S12 as mentioned in VALID_SUBJECTS comment
            continue
            
        try:
            # Convert numeric ID to string format for loading
            subject_str = str(subject_id)
            subject_data = load_subject(subject_str)
            
            # Add subject if we have data
            if 'acc' in subject_data and len(subject_data['acc']) > 0:
                subjects_data[f"S{subject_str}"] = subject_data
                logger.info(f"Added subject S{subject_str} with {len(subject_data['acc'])} samples")
            else:
                logger.warning(f"S{subject_str}: No acceleration data found. Skipping subject.")
                
        except (DataLoadingError, DataValidationError) as e:
            logger.warning(f"Could not load subject S{subject_str}: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error loading S{subject_str}: {str(e)}")
    
    if not subjects_data:
        logger.error("No valid subjects could be loaded!")
    else:
        logger.info(f"Successfully loaded {len(subjects_data)} subjects")
    
    return subjects_data

def analyze_dataset(subjects_data: Dict) -> pd.DataFrame:
    """Generate comprehensive dataset analysis"""
    analysis = []

    for subject, data in subjects_data.items():
        X = data['acc']
        y = data['label']

        # Find unique classes and their distribution
        unique_labels, counts = np.unique(y, return_counts=True)
        label_dist = dict(zip(unique_labels, counts))
        
        # Calculate basic stats
        subject_stats = {
            'subject': subject,
            'samples': X.shape[0],
            'duration_hours': X.shape[0] / (3*60*60),  # 3Hz sampling
            'acc_mean': X.mean(axis=0).tolist(),
            'acc_std': X.std(axis=0).tolist(),
            'num_classes': len(unique_labels),
            'class_distribution': label_dist,
            'missing_values': np.isnan(X).sum()
        }
        analysis.append(subject_stats)

    return pd.DataFrame(analysis)

def plot_sample_data(subject_data: Dict, subject_id: str, n_samples: int = 1000) -> None:
    """Plot sample acceleration data with labels"""
    import matplotlib.pyplot as plt
    import os

    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)

    X = subject_data['acc'][:n_samples]
    y = subject_data['label'][:n_samples]
    time = np.arange(X.shape[0]) / (3*60)  # Minutes

    plt.figure(figsize=(12, 8))
    
    # Plot acceleration data
    plt.subplot(2, 1, 1)
    for i in range(X.shape[1]):
        plt.plot(time, X[:, i], label=f'Axis {i+1}')

    plt.title(f"Sensor Data - {subject_id}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Acceleration")
    plt.legend()
    
    # Plot labels on the same timeline
    plt.subplot(2, 1, 2)
    plt.plot(time, y, 'r-', label='Labels')
    plt.xlabel("Time (minutes)")
    plt.ylabel("Label")
    plt.title(f"Activity Labels - {subject_id}")
    
    plt.tight_layout()
    plt.savefig(f"results/{subject_id}_sensor_plot.png")
    plt.close()

def plot_class_distribution(subjects_data: Dict) -> None:
    """Plot class distribution across all subjects"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import os
    
    # Ensure results directory exists
    os.makedirs("results", exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    # Plot for each subject
    for i, (subject, data) in enumerate(subjects_data.items()):
        y = data['label']
        unique, counts = np.unique(y, return_counts=True)
        
        plt.subplot(4, 4, i+1)
        sns.barplot(x=[str(int(l)) for l in unique], y=counts)
        plt.title(f"{subject} Label Distribution")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.yscale('log')  # Log scale for better visualization
    
    plt.tight_layout()
    plt.savefig("results/class_distribution.png")
    plt.close()

if __name__ == "__main__":
    try:
        if not os.path.exists(DATASET_PATH):
            logger.error(f"Dataset path not found: {DATASET_PATH}")
        else:
            logger.info(f"Loading dataset from {DATASET_PATH}")
            
            dataset = load_wesad_data()
            analysis_df = analyze_dataset(dataset)
            print("\nDataset Analysis:")
            print(analysis_df.to_string())
            
            plot_class_distribution(dataset)
            
            if dataset:
                first_subject = next(iter(dataset))
                plot_sample_data(dataset[first_subject], first_subject)
                
                logger.info(f"Processing {first_subject}")
                X_train, X_test, y_train, y_test = preprocess_data(dataset[first_subject])
                print(f"\nPreprocessed Data Shapes:")
                print(f"Train: {X_train.shape}, {y_train.shape}")
                print(f"Test: {X_test.shape}, {y_test.shape}")
            
    except Exception as e:
        logger.error(f"Critical failure: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())