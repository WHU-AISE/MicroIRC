class IndexMap:

    def __init__(self, train_index, data_index):
        self.train_index = train_index
        self.data_index = data_index

def index_map(train_index_list, index_map_list):
    data_index_list = []
    for train_index in train_index_list:
        for index_map in index_map_list:
            if index_map.train_index == train_index:
                data_index_list.append(index_map.data_index)
                break
    return data_index_list