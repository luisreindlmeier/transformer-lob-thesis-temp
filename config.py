"""LOB Prediction Configuration."""
from typing import Final, List
import torch
import os

DEVICE: Final[str] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Data structure (LOBSTER format)
LEN_ORDER: Final[int] = 6       # [time, event_type, size, price, direction, depth]
LEN_LEVEL: Final[int] = 4       # [ask_price, ask_size, bid_price, bid_size]
N_LOB_LEVELS: Final[int] = 10
LEN_SMOOTH: Final[int] = 10
LOBSTER_HORIZONS: Final[List[int]] = [10, 20, 50, 100]

# Model hyperparameters
SEQ_SIZE: Final[int] = 128
HIDDEN_DIM: Final[int] = 46     # LEN_ORDER (6) + N_LOB_LEVELS * LEN_LEVEL (40)
NUM_LAYERS: Final[int] = 4
NUM_HEADS: Final[int] = 1
IS_SIN_EMB: Final[bool] = True
LR: Final[float] = 0.0001
BATCH_SIZE: Final[int] = 128

# Training
MAX_EPOCHS: Final[int] = 10
PRECISION: Final[int] = 32

# Paths
BASE_DIR: Final[str] = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: Final[str] = os.environ.get("DATA_DIR", os.path.join(BASE_DIR, "data"))
CHECKPOINT_DIR: Final[str] = os.path.join(BASE_DIR, "checkpoints")

# Data split
SPLIT_RATES: Final[List[float]] = [0.85, 0.05, 0.10]
