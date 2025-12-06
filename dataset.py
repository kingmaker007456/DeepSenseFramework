import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config 

# Global variables to store mean and std calculated from the training set
TRAIN_MEAN = 0.0
TRAIN_STD = 1.0

# --- Worker Initialization Function (FIX: Top-level for multiprocessing) ---
def worker_init_fn(worker_id):
    """Sets distinct seeds for workers to ensure different data augmentation noise."""
    worker_seed = Config.SEED + worker_id 
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

# --- Utility Function: Data Augmentation ---
def jittering(data, scale):
    """Adds small random noise (jitter) to the data."""
    noise = np.random.normal(loc=0., scale=scale, size=data.shape)
    return data + noise

# --- Utility Function: Windowing and Label Majority Vote ---
def create_windows(data, labels, window_size, overlap):
    """
    Segments data and finds the majority label for each window.
    """
    timesteps, channels = data.shape
    step = window_size - overlap
    windows = []
    window_labels = []

    num_windows = (timesteps - window_size) // step + 1
    
    for i in range(num_windows):
        start = i * step
        end = start + window_size
        
        if end > timesteps:
            continue
            
        windows.append(data[start:end, :])
        
        label_segment = labels[start:end]
        if label_segment.size > 0:
             bincount = np.bincount(label_segment)
             majority_label = np.argmax(bincount)
             window_labels.append(majority_label)
    
    return np.array(windows, dtype=np.float32), np.array(window_labels, dtype=np.int64)


# --- DeepSense Custom Dataset Class ---
class SensorDataset(Dataset):
    def __init__(self, data_path, is_train=True):
        self.is_train = is_train
        global TRAIN_MEAN, TRAIN_STD
        print(f"Loading and processing data for {'Train' if is_train else 'Test'}...")

        # --- Simulate Loading Realistic Data ---
        N_steps = 50000 if is_train else 15000
        np.random.seed(Config.SEED + (0 if is_train else 1)) 
        self.raw_data = np.random.randn(N_steps, Config.TOTAL_CHANNELS).astype(np.float32)
        
        activity_blocks = np.repeat(np.arange(Config.NUM_CLASSES), N_steps // Config.NUM_CLASSES)
        self.raw_labels = np.resize(activity_blocks, N_steps)
        np.random.shuffle(self.raw_labels) 
        
        # 1. Normalization
        if self.is_train:
            TRAIN_MEAN = np.mean(self.raw_data, axis=0)
            TRAIN_STD = np.std(self.raw_data, axis=0)
            print("Calculated training set statistics for normalization.")
        
        self.raw_data = (self.raw_data - TRAIN_MEAN) / (TRAIN_STD + 1e-6)
        
        # 2. Windowing Data and Labels with Majority Vote
        self.windows, self.labels = create_windows(
            self.raw_data, 
            self.raw_labels, 
            Config.WINDOW_SIZE, 
            Config.OVERLAP
        )
        
        # NOTE: Using torch.from_numpy retains the numpy float32 type
        self.windows_tensor = torch.from_numpy(self.windows)
        self.labels_tensor = torch.from_numpy(self.labels)

        print(f"Data ready. Windows shape: {self.windows_tensor.shape}, Labels shape: {self.labels_tensor.shape}")

    def __len__(self):
        return len(self.labels_tensor)

    def __getitem__(self, idx):
        window = self.windows_tensor[idx]
        label = self.labels_tensor[idx]
        
        # Apply data augmentation only to the training set
        if self.is_train and Config.JITTER_SCALE > 0:
            # Note: Jittering relies on numpy and then converts back to torch
            window_np = window.numpy()
            window_augmented = jittering(window_np, Config.JITTER_SCALE)
            window = torch.from_numpy(window_augmented)

        return window, label

# --- Helper function for quick DataLoader creation ---
def get_dataloaders():
    train_dataset = SensorDataset("data/train", is_train=True)
    test_dataset = SensorDataset("data/test", is_train=False)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=True, 
        drop_last=True,
        num_workers=4, 
        # FIX: Conditionally set pin_memory to suppress the warning on CPU
        pin_memory=Config.PIN_MEMORY, 
        worker_init_fn=worker_init_fn 
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=Config.BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        # FIX: Conditionally set pin_memory
        pin_memory=Config.PIN_MEMORY,
        worker_init_fn=worker_init_fn
    )
    
    return train_loader, test_loader
