DeepSense Attention for Human Activity Recognition (HAR)
This repository contains a PyTorch implementation of a DeepSense-like architecture enhanced with a self-attention mechanism, designed for temporal sequence processing in Human Activity Recognition (HAR) tasks using multi-sensor data (e.g., Accelerometer and Gyroscope).

🚀 Key Features
DeepSense Architecture: Utilizes parallel 1D Convolutional Neural Networks (CNNs) for extracting low-level features from different sensor modalities.

Sequential Modeling: Features are passed through a Bidirectional Long Short-Term Memory (Bi-LSTM) network to capture temporal dependencies.

Attention Mechanism: Incorporates a self-attention layer on the LSTM outputs to weigh the importance of different time steps before final classification.

Robust Training: Includes Early Stopping, Learning Rate Scheduling, and Gradient Clipping for stable and efficient optimization.

Visualization: Integration with TensorBoard for real-time monitoring of loss, accuracy, F1-scores, and plotting the Confusion Matrix and Classification Report.

Data Handling: Uses custom PyTorch Dataset and DataLoader with multiprocessing, data augmentation (Jittering), and majority-vote windowing for robust data preparation.

📦 Project Structure
.
├── config.py           # Project configuration (hyperparameters, paths, device settings)
├── dataset.py          # Custom PyTorch Dataset and DataLoader for data loading, windowing, and augmentation
├── model.py            # DeepSenseAttention model definition (CNN, Bi-LSTM, Attention, Classifier)
├── train.py            # Main training and evaluation script with logging, saving, and early stopping
├── README.md           # This file
├── runs/               # TensorBoard logs are stored here (Config.LOG_DIR)
└── best_model.pth      # Best model weights are saved here (Config.MODEL_SAVE_PATH)
🛠️ Prerequisites
Ensure you have Python (3.7+) installed. This project relies on the following major libraries:

Bash

# Recommended environment setup
conda create -n deepsense_har python=3.10
conda activate deepsense_har

# Install PyTorch and necessary packages
pip install torch torchvision torchaudio numpy scikit-learn tqdm matplotlib seaborn tensorboard
⚙️ Configuration (config.py)
All hyperparameter tuning and resource settings are managed in config.py. Key parameters to check before training include:

Parameter	Description	Default Value
DEVICE	Target device: cuda if available, otherwise cpu.	torch.device(...)
PIN_MEMORY	Automatically set to True if cuda is available, False otherwise.	True or False
WINDOW_SIZE	Number of time steps per input segment.	128
OVERLAP	Overlap between consecutive windows.	64
NUM_CLASSES	Number of activity classes to classify.	5
BATCH_SIZE	Training and testing batch size.	64
NUM_EPOCHS	Maximum number of training epochs.	50
JITTER_SCALE	Standard deviation for noise added during data augmentation.	0.01
EARLY_STOPPING_PATIENCE	Epochs to wait for F1 improvement before stopping.	10

Export to Sheets

🏃 Getting Started
1. Model Architecture
The DeepSense Attention model is structured to process multi-channel time-series data:

Sensor Segmentation: Input sequence (Window_size, Channels) is split into modalities (e.g., Accelerometer and Gyroscope).

Feature Extraction: Each modality is processed by a dedicated TemporalCNN (a 1D CNN block) to extract high-level feature vectors.

Fusion & Sequence Learning: Features are concatenated and fed into a Bi-LSTM to learn sequential dependencies across the window.

Attention: A self-attention block weighs the significance of the LSTM outputs.

Classification: The final context vector is passed to a dense classifier to predict the activity label.

2. Training
To start the training process, simply run the main script:

Bash

python train.py
The script will:

Initialize the model and data loaders.

Begin training, showing progress via tqdm.

Print detailed metrics (Loss, Accuracy, F1-Score, Classification Report, Confusion Matrix) after each epoch.

Save the best model based on the validation F1-score to best_deepsense_attention_model.pth.

3. Monitoring with TensorBoard
During or after training, you can monitor the performance metrics and visualizations (Confusion Matrix, Classification Report) using TensorBoard:

Bash

tensorboard --logdir=runs/deepsense_attention_logs
Open your browser and navigate to the address shown in the terminal (usually http://localhost:6006/).

🐛 Troubleshooting
This robust implementation includes fixes for common PyTorch and Python multiprocessing issues:

Error/Warning	Cause	Solution Implemented
AttributeError: Can't get local object 'get_dataloaders.<locals>.worker_init_fn'	Local function definition inside get_dataloaders cannot be pickled by Windows multiprocessing ('spawn').	dataset.py: Moved worker_init_fn to the top level of the module.
RuntimeError: Input type (double) and bias type (float) should be the same	Data loaded as float64 but model parameters are float32.	train.py: Explicitly cast input tensor to float32 using .float() inside the train and evaluate loops.
UserWarning: 'pin_memory' argument is set as true but no accelerator is found...	pin_memory=True is redundant when running on CPU.	config.py & dataset.py: Conditionally set PIN_MEMORY=False when Config.DEVICE is 'cpu'.
Exception ignored in: <function Image.__del__ at ...>	Matplotlib figure garbage collection conflict with TensorBoard logging.	train.py: Explicitly called plt.close(fig) after logging the confusion matrix to release resources.