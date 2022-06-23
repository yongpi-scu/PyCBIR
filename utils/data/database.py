from utils.data import DataLoader
from descriptors.color import ColorHistogram
from descriptors.local import SIFT

class Database:
    def __init__(self, data_dir="dataset/", descriptor_method="ColorHistogram", indexing="bruteforce", img_mode="RGB"):
        # offline extract features
        dataloader = DataLoader(data_dir=data_dir,img_mode=img_mode)
        descriptor_func = globals()[descriptor_method]()
        features = descriptor_func.extract_batch(dataloader)
        
        # indexing methods for accelerate search process
        if indexing == "lsh":
            pass
        elif indexing == "flsh":
            pass
        elif indexing == "bow":
            pass
        elif indexing == "bruteforce":
            features = features
        else:
            raise NotImplementedError(
                "indexing method %s not implemented!" % indexing)
        self.features = features
        self.dataloader = dataloader
    
    def get_class(self):
        return self.dataloader.get_class()
    
    def __len__(self):
        return len(self.features)
    
    def __iter__(self):
        self.index=0
        return self
    
    def __next__(self):
        if self.index<len(self.features):
            feature  = self.features[self.index]
            self.index+=1
            return feature
        else:
            raise StopIteration