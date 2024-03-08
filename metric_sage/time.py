class Time:

    def __init__(self, simple, begin_time, end_time, root_cause, root_cause_level, failure_type, label, index):
        self.simple = simple
        self.begin = begin_time
        self.end = end_time
        self.begin_index = -1
        self.end_index = -1
        self.count = 0
        self.root_cause = root_cause
        self.root_cause_level = root_cause_level
        self.failure_type = failure_type
        self.label = label
        self.index = index

    def in_time(self, time, index):
        is_in_time = False
        if time >= self.begin and time < self.end:
            if self.begin_index == -1:
                self.begin_index = index
                self.end_index = index
            if index > self.end_index:
                self.end_index = index
            is_in_time = True
            self.count = self.end_index - self.begin_index + 1
        return is_in_time