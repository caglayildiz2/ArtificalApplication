import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = '0'

import tensorflow as tf
gpus = tf.config.list_physical_devices("GPU")
try:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
except:
    pass

import cv2
import tqdm
import pickle
import numpy as np
import matplotlib.pyplot as plt

from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .Timer import Timer

class DatasetHandler:
    def __init__(self, is_training, dataset_path, sequence_length, resize_width, resize_height, color_channel_count, validation_ratio, test_ratio):
        if is_training:
            self.DATASET_PATH = dataset_path
            self.SEQUENCE_LENGTH = sequence_length
            self.RESIZE_WIDTH = resize_width
            self.RESIZE_HEIGHT = resize_height
            self.COLOR_CHANNEL_COUNT = color_channel_count
            self.VALIDATION_RATIO = validation_ratio
            self.TEST_RATIO = test_ratio
        
            self.timer = Timer()
            
        else:
            self.SEQUENCE_LENGTH = sequence_length
            self.RESIZE_WIDTH = resize_width
            self.RESIZE_HEIGHT = resize_height
            self.COLOR_CHANNEL_COUNT = color_channel_count
            
        return

    def init(self):
        self.labeled_video_paths, self.labels, self.videos, self.included_video_paths = self.read_dataset()
        
        self.videos = np.asarray(self.videos)
        
        label_encoder = LabelEncoder()
        self.labels = label_encoder.fit_transform(self.labels)
        self.labels = to_categorical(self.labels, num_classes=len(self.labeled_video_paths))
        
        self.X_train, self.X_test, self.Y_train, self.Y_test = train_test_split(self.videos, self.labels, test_size=self.TEST_RATIO, shuffle=True)
        self.X_train, self.X_validation, self.Y_train, self.Y_validation = train_test_split(self.X_train, self.Y_train, test_size=self.VALIDATION_RATIO, shuffle=True)
        
        self.X_train = np.asarray(self.X_train)
        self.X_validation = np.asarray(self.X_validation)
        self.X_test = np.asarray(self.X_test)
        self.Y_train = np.asarray(self.Y_train)
        self.Y_validation = np.asarray(self.Y_validation)
        self.Y_test = np.asarray(self.Y_test)

        return
    
        
    def read_dataset(self):
        if os.path.exists('preparation/tagged_paths.pkl') and os.path.exists('preparation/labels.pkl') \
            and os.path.exists('preparation/videos.pkl') and os.path.exists('preparation/included_video_paths.pkl'):
            
            print("Loading dataset from saved files...", end="\n\n")
            
            with open('preparation/tagged_paths.pkl', 'rb') as f:
                tagged_paths = pickle.load(f)
                
            with open('preparation/included_video_paths.pkl', 'rb') as f:
                included_video_paths = pickle.load(f)
                
            with open('preparation/labels.pkl', 'rb') as f:
                labels = pickle.load(f)
                
            with open('preparation/videos.pkl', 'rb') as f:
                video_frames = pickle.load(f)
                            
            print("Dataset loaded successfully.", end="\n\n")
            
            return tagged_paths, labels, video_frames, included_video_paths
        
        tagged_paths = {}
        categories = os.listdir(self.DATASET_PATH)
        for category in categories:
            if category not in tagged_paths:
                tagged_paths[category] = []
                
            folder_path = os.path.join(self.DATASET_PATH, category)
            video_list = os.listdir(folder_path)
            for video in video_list:
                video_path = os.path.join(folder_path, video)
                tagged_paths[category].append(tf.convert_to_tensor(video_path))
                
        labels = []
        video_frames = []
        included_video_paths = []
        
        _ = self.get_mean_std_frame(np.zeros((self.SEQUENCE_LENGTH, self.RESIZE_HEIGHT, self.RESIZE_WIDTH, 1)))
        print("\nPlease wait...", end="\n\n")
        
        self.timer.start()
        total_videos = sum([len(v) for k, v in tagged_paths.items()])
        with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]",
                       desc="Saving videos...", unit=" videos", total=total_videos, leave=False) as pbar_full:
            label_count = len(tagged_paths)
            
            with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                            desc="Reading folders...", unit=" folders", total=label_count, leave=False) as pbar_folders:
                for category_index, (category, video_path_list) in enumerate(tagged_paths.items()):
                    pbar_folders.set_description(f"Reading folder: \"{category}\"")
                    
                    video_count = len(video_path_list)
                    with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                                   desc="Reading videos...", unit= " videos", total=video_count, leave=False) as pbar_videos:
                        for video_path in video_path_list:
                            decoded_video_path = video_path.numpy().decode("utf-8")
                            dataset_tag, main_folder, category, video_name = decoded_video_path.split('\\')
                            pbar_videos.set_description(f"Reading video: \"{video_name}\"")
                            
                            current_video_frames = self.read_video(decoded_video_path)
                            if len(current_video_frames) == self.SEQUENCE_LENGTH:
                                labels.append(category_index)
                                video_frames.append(current_video_frames)
                                included_video_paths.append(video_path)
                            
                            pbar_videos.update(1)
                            pbar_full.update(1)
                            
                    pbar_folders.update(1)
                    
        self.timer.stop()
        print("Dataset read.")
        
        time_elapsed = self.timer.get_formatted_time()
        print(f"Dataset read in {time_elapsed}.", end="\n\n")
        
        if not os.path.exists('preparation'):
            os.mkdir('preparation')
            
        with open('preparation/tagged_paths.pkl', 'wb') as f:
            pickle.dump(tagged_paths, f, protocol=4)

        with open('preparation/included_video_paths.pkl', 'wb') as f:
            pickle.dump(included_video_paths, f, protocol=4)
            
        with open('preparation/labels.pkl', 'wb') as f:
            pickle.dump(labels, f, protocol=4)
            
        with open('preparation/videos.pkl', 'wb') as f:
            pickle.dump(video_frames, f, protocol=4)
            
        return tagged_paths, labels, video_frames, included_video_paths


    def read_video(self, video_path):
        video_reader = cv2.VideoCapture(video_path)
        frames = []
        if not video_reader.isOpened():
            print("Error: Could not open video.")
            return
        
        frame_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        skip_frames_window = max(int(frame_count / self.SEQUENCE_LENGTH), 1)
        
        with tqdm.tqdm(bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_noinv_fmt}]", 
                       desc="Reading sequence frames...", unit=" frames", total=self.SEQUENCE_LENGTH, leave=False) as pbar_frames:
            for frame_index in range(self.SEQUENCE_LENGTH):
                video_reader.set(cv2.CAP_PROP_POS_FRAMES, (frame_index * skip_frames_window))
                
                success, frame = video_reader.read()
            
                if frame is None:
                    break
            
                if not success:
                    print("Error: Could not read frame.")
                    break
                
                processed_frame = self.process_frame(frame)
                frames.append(processed_frame)
                pbar_frames.update(1)
                
        video_reader.release()
        return self.get_mean_std_frame(frames)
    
    def get_mean_std_frame(self, frames):
        mean = tf.math.reduce_mean(frames)
        std = tf.math.reduce_std(tf.cast(frames, tf.float32))
        return tf.cast((frames - mean), tf.float32) / std
    
    def process_frame(self, frame):
        resized_frame = tf.image.resize(frame, [self.RESIZE_HEIGHT, self.RESIZE_WIDTH])
        gray_scaled_image = tf.image.rgb_to_grayscale(resized_frame)
        gray_scaled_image = tf.cast(gray_scaled_image, tf.float32)
        normalized_frame = gray_scaled_image / 255.0
        
        return normalized_frame
    