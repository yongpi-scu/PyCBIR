from utils.data import DataLoader
from descriptors.color import ColorHistogram
from descriptors.local import SIFT
from indexing.lsh import LSH
from indexing.flsh import FLSH
from indexing.linear import Linear
from indexing.bow import BOW


class Database:
    def __init__(self, dataloader, descriptor_method="ColorHistogram", indexing_method="lsh", index_dim=32):
        # offline extract features
        self.descriptor_func = globals()[descriptor_method]()
        self.descriptors = self.descriptor_func.extract_batch(dataloader)
        
        self.dataloader = dataloader
        self.indexing_method = indexing_method
        self.discriptor_dim = self.descriptors[0]["descriptor"].shape[-1]
        
        self.index_dim = index_dim
        # init
        self.__init_indexHub()

    
    def __init_indexHub(self):
        # indexing methods for accelerate search process
        if self.indexing_method == "lsh":
            self.indexHub = LSH(self.discriptor_dim, self.index_dim)
        elif self.indexing_method == "flsh":
            self.indexHub = FLSH(self.discriptor_dim, self.index_dim)
        elif self.indexing_method == "bow+k64+lsh":
            k = int(self.indexing_method.split("k")[1].split("+")[0])
            self.bow_model = BOW(self.descriptors, k=k, batch_size=1000) 
            # import pdb;pdb.set_trace()
            for sample, tfidf in zip(self.descriptors,self.bow_model.get_features()):
                sample["descriptor"]=tfidf
            
            self.indexHub = LSH(k, self.index_dim)
            
        elif self.indexing_method in ["bruteforce", "linear"]:
            self.indexHub = Linear()
        else:
            raise NotImplementedError(
                "indexing method %s not implemented!" % self.indexing_method)
        
        for descriptor in self.descriptors:
            self.indexHub.insert(descriptor)
    
    def query(self, sample, depth=3, dist_func="l0"):
        sample["descriptor"] = self.descriptor_func.extract_single(sample["img_array"])
        print("shape1",sample["descriptor"].shape)
        if hasattr(self, "bow_model"):
            sample["descriptor"] = self.bow_model.convert(sample)
        print("shape2",sample["descriptor"].shape)
        results = self.indexHub.indexing(sample, depth, dist_func)
        return results
    
    def insert(self, sample):
        sample["descriptor"] = self.descriptor_func.extract_single(sample["img_array"])
        self.indexHub.insert(sample)
    
    def get_class(self):
        return self.dataloader.get_class()
    
    def __len__(self):
        return len(self.features)
    
    def __iter__(self):
        self.index=0
        return self
    
    def __next__(self):
        if self.index<len(self.features):
            feature = self.features[self.index]
            self.index+=1
            return feature
        else:
            raise StopIteration