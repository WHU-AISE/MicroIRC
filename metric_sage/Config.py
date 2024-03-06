import time
from enum import Enum


class Config:
    def __init__(self, is_train, time_window_minutes=10, sample_interval=1 / 12, num_sample=10):
        self.is_train = is_train
        self.time_window_minutes = time_window_minutes
        self.sample_interval = sample_interval
        self.start = int(round((time.time() - self.time_window_minutes * 60)))
        self.end = int(round(time.time()))
        self.num_sample = num_sample
