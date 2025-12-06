import torch

# --- GENERAL CONFIGURATION ---
class Config:
    """
    Configuration settings for the DeepSense Attention model,
    covering data, model architecture, and training hyperparameters.
    """
    # Environment Setup
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SEED = 42
    
    # NEW: Control pin_memory based on device availability
    # It is beneficial to keep it True if a GPU (cuda) is detected.
    PIN_MEMORY = torch.cuda.is_available() 

    # Data Parameters (Simulated for HAR)
    WINDOW_SIZE = 128      # Timesteps per window
    OVERLAP = 64           # Overlap between windows (50%)
    TOTAL_CHANNELS = 6     # e.g., Accel(3) + Gyro(3)
    ACCEL_CHANNELS = 3
    GYRO_CHANNELS = 3
    NUM_CLASSES = 5        # e.g., Walking, Running, Sitting, Standing, Cycling
    
    # Data Augmentation Parameters (New Enhancement)
    JITTER_SCALE = 0.01    # Standard deviation for jittering noise

    # Model Parameters
    CNN_OUT_CHANNELS = 64
    LSTM_HIDDEN_SIZE = 128
    LSTM_NUM_LAYERS = 2
    DROPOUT_RATE = 0.3
    
    # Attention Parameters 
    ATTENTION_HEADS = 4
    FFN_HIDDEN_DIM = 256 

    # Training Parameters
    BATCH_SIZE = 64
    NUM_EPOCHS = 50        
    LEARNING_RATE = 0.001
    LR_SCHEDULER_STEP = 5  
    LR_GAMMA = 0.1         

    # Optimizer Parameters 
    ADAM_WEIGHT_DECAY = 1e-4 

    # Gradient Clipping
    GRADIENT_CLIP_VAL = 1.0 

    # Early Stopping 
    EARLY_STOPPING_PATIENCE = 10 
    MIN_DELTA = 0.0001           

    # Output Paths
    MODEL_SAVE_PATH = "best_deepsense_attention_model.pth"
    LOG_DIR = "runs/deepsense_attention_logs" 
    
    # Weight Initialization 
    INIT_METHOD = "kaiming_uniform" # Options: kaiming_uniform, xavier_uniform, none

# Helper to set seed for reproducibility
torch.manual_seed(Config.SEED)
if Config.DEVICE.type == 'cuda':
    torch.cuda.manual_seed(Config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
