from torchvision import transforms
from torch.utils.data import Dataset


class TrainDataset(Dataset):
    def __init__(self, x_train, y_train):
        super(TrainDataset, self).__init__()
        self.transforms = transforms.ToTensor()
        self.x = x_train
        self.y = y_train
        
    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        return x, y
    
    def __len__(self):
        return len(self.x)


class TestDataset(Dataset):
    def __init__(self, x_test, y_test):
        super(TestDataset, self).__init__()
        self.x = x_test
        self.y = y_test
        
    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        return x, y
    
    def __len__(self):
        return len(self.x)

class DownstreamDataset(Dataset):
    def __init__(self, x_test, y_test):
        super(DownstreamDataset, self).__init__()
        self.x = x_test
        self.y = y_test
        
    def __getitem__(self, index):
        x = self.x[index, ...]
        y = self.y[index, ...]
        return x, y
    
    def __len__(self):
        return len(self.x)
