import os

from structures import LSTM_Model

# Define settings for dataset, model, compiling, training, and statistics
def train_lstm_model():
    settings = {
        "dataset": {
            "dataset_path": os.path.join("dataset", "Fit3D Video Dataset"),
            "sequence_length": 75,
            "resize_width": 128,
            "resize_height": 128,
            "color_channels": 1,
            "validation_ratio": 0.2,
            "test_ratio": 0.2
        },
        
        "model": {
            "filter_counts": [32, 64, 128, 256],
            "kernel_sizes": [(3, 3, 3), (3, 3, 3), (3, 3, 3), (3, 3, 3)],
            "activations": ["relu", "relu", "relu", "relu", "relu", "softmax"],
            "pool_sizes": [(2, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)],
            "lstm_unit_counts": [128],
            "dense_unit_counts": [512],
            "dropout_unit_counts": [0.5]
        },
        
        "compiling": {
            "scheduler_epoch_patience": 30,
            "scheduler_learning_rate_multiplier": -0.1,
            "early_stoppping_monitor": "val_loss",
            "early_stopping_patience": 10,
            "early_stopping_mode_restore_best_weights": True,
            "learning_rate": 0.0001,
            "loss": "categorical_crossentropy",
            "metrics": ["accuracy"]
        },

        "training": {
            "epoch_count": 100,
            "batch_size": 8,
            "shuffle": True
        },
        
        "statistics": {
            "graph_size": (12, 6)
        }
    }
    
    lstm = LSTM_Model(settings)
    lstm.execute()
    
    return


def main():
    train_lstm_model()
    
    return 0


if __name__ == "__main__":
    main()
    