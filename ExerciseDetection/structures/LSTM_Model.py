import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
except:
    pass

from keras.models import Sequential
from keras.layers import Conv3D, LSTM, Dense, Dropout, Bidirectional, MaxPooling3D, Activation, Reshape, SpatialDropout3D, BatchNormalization, TimeDistributed, Flatten
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.utils import plot_model

import matplotlib.pyplot as plt
import datetime as dt
import numpy as np
import shutil

from .LSTM_ModelSettings import LSTM_ModelSettings
from .DatasetHandler import DatasetHandler
from .Timer import Timer

class LSTM_Model:
    def __init__(self, settings):
        self.settings = settings
        self.timer = Timer()
        
        return
    
    def execute(self):
        self.init()
        self.train()
        self.test()
        self.show_statistics()
        self.save()


    def init(self):
        self.all_settings = LSTM_ModelSettings(dataset_path=self.settings["dataset"]["dataset_path"],
                                               dataset_sequence_length=self.settings["dataset"]["sequence_length"],
                                               dataset_resize_width=self.settings["dataset"]["resize_width"],
                                               dataset_resize_height=self.settings["dataset"]["resize_height"],
                                               dataset_color_channels=self.settings["dataset"]["color_channels"],
                                               dataset_validation_ratio=self.settings["dataset"]["validation_ratio"],
                                               dataset_test_ratio=self.settings["dataset"]["test_ratio"],
                                               
                                               model_filter_counts=self.settings["model"]["filter_counts"],
                                               model_kernel_sizes=self.settings["model"]["kernel_sizes"],
                                               model_activations=self.settings["model"]["activations"],
                                               model_pool_sizes=self.settings["model"]["pool_sizes"],
                                               model_lstm_unit_counts=self.settings["model"]["lstm_unit_counts"],
                                               model_dense_unit_counts=self.settings["model"]["dense_unit_counts"],
                                               model_dropout_unit_counts=self.settings["model"]["dropout_unit_counts"],
                                               
                                               compiling_scheduler_epoch_patience=self.settings["compiling"]["scheduler_epoch_patience"],
                                               compiling_scheduler_learning_rate_multiplier=self.settings["compiling"]["scheduler_learning_rate_multiplier"],
                                               compiling_early_stopping_monitor=self.settings["compiling"]["early_stoppping_monitor"],
                                               compiling_early_stopping_patience=self.settings["compiling"]["early_stopping_patience"],
                                               compiling_early_stopping_mode_restore_best_weights=self.settings["compiling"]["early_stopping_mode_restore_best_weights"],
                                               compiling_learning_rate=self.settings["compiling"]["learning_rate"],
                                               compiling_loss=self.settings["compiling"]["loss"],
                                               compiling_metrics=self.settings["compiling"]["metrics"],
                                               
                                               training_epoch_count=self.settings["training"]["epoch_count"],
                                               training_batch_size=self.settings["training"]["batch_size"],
                                               training_shuffle=self.settings["training"]["shuffle"],
                                               
                                               statistics_graph_size=self.settings["statistics"]["graph_size"])
        
        self.dataset_handler = DatasetHandler(True,
                                              self.all_settings.get_dataset_path(),
                                              self.all_settings.get_dataset_sequence_length(),
                                              self.all_settings.get_dataset_resize_width(),
                                              self.all_settings.get_dataset_resize_height(),
                                              self.all_settings.get_dataset_color_channel_count(),
                                              self.all_settings.get_dataset_validation_ratio(),
                                              self.all_settings.get_dataset_test_ratio())
        self.dataset_handler.init()
        
        self.model = self.create_model()
        self.training_callbacks = self.create_callbacks()
        self.model.compile(optimizer=Adam(self.all_settings.get_compiling_learning_rate()),
                           loss=self.all_settings.get_compiling_loss(),
                           metrics=self.all_settings.get_compiling_metrics())
        
        return
    
    def train(self):
        device = ""
        if len(gpus) != 0:
            device = "/device:GPU:0"
            
        else:
            device = "/device:CPU:0"
            
        print("Please wait while the program trains the model...", end="\n\n")
        
        self.timer.start()
        with tf.device(device):
            self.training_history = self.model.fit(x=self.dataset_handler.X_train,
                                                   y=self.dataset_handler.Y_train,
                                                   epochs=self.all_settings.get_training_epoch_count(),
                                                   batch_size=self.all_settings.get_training_batch_size(),
                                                   shuffle=self.all_settings.get_training_shuffle(),
                                                   validation_data=(self.dataset_handler.X_validation, self.dataset_handler.Y_validation),
                                                   callbacks=self.training_callbacks)
        self.timer.stop()
        print("Model trained successfully.")
        
        time_elapsed = self.timer.get_formatted_time()
        print(f"Model Trained in {time_elapsed}.", end="\n\n")

        return
    
    def test(self):
        print("Please wait while the program tests the model...", end="\n\n")
        
        self.timer.start()
        self.evaluation_history = self.model.evaluate(self.dataset_handler.X_test, self.dataset_handler.Y_test)
        self.timer.stop()
        print("Model tested successfully.")
        
        time_elapsed = self.timer.get_formatted_time()
        print(f"Model Tested in {time_elapsed}.", end="\n\n")
        
        return

    def show_statistics(self):
        accuracy_history = self.training_history.history["accuracy"]
        val_accuracy_history = self.training_history.history["val_accuracy"]
        accuracy_history_indexes = range(len(accuracy_history))
        
        loss_history = self.training_history.history["loss"]
        val_loss_history = self.training_history.history["val_loss"]
        loss_history_indexes = range(len(loss_history))
        
        plt.figure(figsize=self.all_settings.get_statistics_graph_size())
        plt.gcf().canvas.set_window_title("Accuracy Statistics Over Time")
        plt.title("Accuracy Statistics Over Time")
        plt.plot(accuracy_history_indexes, accuracy_history, color="blue", label="Accuracy")
        plt.plot(accuracy_history_indexes, val_accuracy_history, color="red", label="Validation Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy Value")
        plt.legend()
        
        plt.figure(figsize=self.all_settings.get_statistics_graph_size())
        plt.gcf().canvas.set_window_title("Loss Statistics Over Time")
        plt.title("Loss Statistics Over Time")
        plt.plot(loss_history_indexes, loss_history, color="blue", label="Loss")
        plt.plot(loss_history_indexes, val_loss_history, color="red", label="Validation Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss Value")
        plt.legend()
        
        plt.show()
        
        return
    
    def save(self):
        loss, accuracy = self.evaluation_history
        print("Accuracy: {:.2f}, Loss: {:.2f}%".format((accuracy * 100), (loss * 100)))

        time_format = "%d_%m_%Y__%H_%M_%S"
        current_date_time = dt.datetime.now()
        date_string = dt.datetime.strftime(current_date_time, time_format)
        
        model_file_name = f"LRCNModel_{date_string}__Accuracy_{accuracy}__Loss_{loss}"
        model_save_folder_path = os.path.join("models", "saves")
        if not os.path.exists(model_save_folder_path):
            os.makedirs(model_save_folder_path)

        model_save_path = os.path.join(model_save_folder_path, (model_file_name + ".h5"))
        
        model_plot_folder = os.path.join("models", "plots")
        if not os.path.exists(model_plot_folder):
            os.makedirs(model_plot_folder)
            
        model_plot_path = os.path.join(model_plot_folder, (model_file_name + ".png"))
        
        print("Please wait while the program saves the model...", end="\n\n")
        
        self.timer.start()
        self.model.save(model_save_path)
        plot_model(self.model, to_file=model_plot_path, show_shapes=True, show_layer_names=True)
        self.timer.stop()
        print("Model saved successfully.")
        
        time_elapsed = self.timer.get_formatted_time()
        print(f"Model Saved in {time_elapsed}.")
        
        exact_model_save_path = os.path.abspath(model_save_path)
        exact_model_plot_path = os.path.abspath(model_plot_path)
        print(f"Model File Saved to {exact_model_save_path}")
        print(f"Model Plot Saved to {exact_model_plot_path}", end="\n\n")
        
        option = input("Do you want to replace this file with model.h5? (Y/N): ")
        if option.lower() == "y":
            if os.path.exists("model.h5"):
                os.remove("model.h5")
            
            shutil.copy(exact_model_save_path, "model.h5")
                
        elif option.lower() == 'n':
            return

        else:
            print("Invalid option. Model is not replaced. Please do it manually.", end="\n\n")
        
        return
    

    def create_model(self):
        model = Sequential()
        
        model.add(Conv3D(filters=self.all_settings.get_model_filter_count_for_layer(0),
                         kernel_size=self.all_settings.get_model_kernel_size_for_layer(0),
                         activation=self.all_settings.get_model_activation_for_layer(0),
                         input_shape=(self.dataset_handler.SEQUENCE_LENGTH,
                                      self.dataset_handler.RESIZE_WIDTH,
                                      self.dataset_handler.RESIZE_HEIGHT,
                                      self.dataset_handler.COLOR_CHANNEL_COUNT)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_model_pool_size_for_layer(0)))

        model.add(Conv3D(filters=self.all_settings.get_model_filter_count_for_layer(1),
                         kernel_size=self.all_settings.get_model_kernel_size_for_layer(1),
                         activation=self.all_settings.get_model_activation_for_layer(1)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_model_pool_size_for_layer(1)))
        
        model.add(Conv3D(filters=self.all_settings.get_model_filter_count_for_layer(2),
                         kernel_size=self.all_settings.get_model_kernel_size_for_layer(2),
                         activation=self.all_settings.get_model_activation_for_layer(2)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_model_pool_size_for_layer(2)))
        
        model.add(Conv3D(filters=self.all_settings.get_model_filter_count_for_layer(3),
                         kernel_size=self.all_settings.get_model_kernel_size_for_layer(3),
                         activation=self.all_settings.get_model_activation_for_layer(3)))
        
        model.add(MaxPooling3D(pool_size=self.all_settings.get_model_pool_size_for_layer(3)))
        
        model.add(TimeDistributed(Flatten()))
        
        model.add(LSTM(units=self.all_settings.get_model_lstm_unit_count_for_layer(0)))
        
        model.add(Dense(units=self.all_settings.get_model_dense_unit_count_for_layer(0),
                        activation=self.all_settings.get_model_activation_for_layer(4)))
        
        model.add(Dropout(rate=self.all_settings.get_model_dropout_unit_count_for_layer(0)))

        model.add(Dense(units=len(self.dataset_handler.labeled_video_paths),
                        activation=self.all_settings.get_model_activation_for_layer(5)))
        
        model.summary()
        return model

    def create_callbacks(self):
        scheduler_callback = LearningRateScheduler(self.scheduler)
        
        early_stopping_callback = EarlyStopping(monitor=self.all_settings.get_compiling_early_stopping_monitor(),
                                                patience=self.all_settings.get_compiling_early_stopping_patience(),
                                                restore_best_weights=self.all_settings.get_compiling_early_stopping_mode_restore_best_weights())
        
        return [scheduler_callback, early_stopping_callback]
    
    def scheduler(self, epoch, lr):
        if epoch < self.all_settings.get_compiling_scheduler_epoch_patience():
            return lr
        
        else:
            return lr * tf.math.exp(self.all_settings.get_compiling_scheduler_learning_rate_multiplier())
    