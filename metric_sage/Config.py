import time
import torch


class Config:
    def __init__(self, is_train=True, time_window_minutes=10,
                 sample_interval=1 / 12, num_sample=10, batch_size=120,
                 epochs=10000, learning_rate=0.01, minute=10, alpha=0.8,
                 instance_tolerant=0.01, service_tolerant=0.03, candidate_count=10,
                 rate=1, epoch_size=1, train_rate=0.8, test_rate=0.1):
        self.is_train = is_train
        self.time_window_minutes = time_window_minutes
        self.sample_interval = sample_interval
        self.start = int(round((time.time() - self.time_window_minutes * 60)))
        self.end = int(round(time.time()))
        self.num_sample = num_sample
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.minute = minute
        self.alpha = alpha
        self.instance_tolerant = instance_tolerant
        self.service_tolerant = service_tolerant
        self.candidate_count = candidate_count
        # rate=1 means training all anomaly types, you can set 0 < rate <= 1, e.g., {0.8, 0.6, 0.4} mentioned in paper
        self.rate = rate
        self.node_overflow = int(minute / sample_interval / 2)
        self.cuda = torch.cuda.is_available()
        self.epoch_size = epoch_size
        self.train_rate = train_rate
        self.test_rate = test_rate
        self.root_cause_service_2_columns = None
        self.combine_columns_index_map = None
