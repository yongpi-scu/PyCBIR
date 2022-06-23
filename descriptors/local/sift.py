import itertools
import os
import pickle
import numpy as np
import cv2

class SIFT:
    def __init__(self, nfeatures=1, cache_dir="cache", normalize=True) -> None:
        
        self.nfeatures = nfeatures  # histogram bins
        self.normalize = normalize  # distance type
        self.cache_dir = cache_dir
        self.sift_extractor = cv2.SIFT_create(nfeatures=nfeatures)
        os.makedirs(self.cache_dir, exist_ok=True)

    def extract_batch(self, dataloader, verbose=True):
        descriptor_cache = "SIFT_cache-n_keypoints{}".format(self.nfeatures)
        try:
            descriptors = pickle.load(
                open(os.path.join(self.cache_dir, descriptor_cache), "rb", True))
            if verbose:
                print("Using cache..., config=%s" %
                      (descriptor_cache))
        except:
            if verbose:
                print("Extracting features..., config=%s" % (
                    descriptor_cache))
            descriptors = []
            for idx, sample in enumerate(dataloader):
                img, cls = sample["img_array"], sample["cls"]
                if verbose:
                    print("%d/%d, %s"%(idx,len(dataloader),sample["img_name"]))
                descriptor = self.extract_single(img)
                descriptors.append({
                                'img_name':  sample["img_name"],
                                'cls':  cls,
                                'descriptor': descriptor
                    })
            pickle.dump(descriptors, open(os.path.join(self.cache_dir, descriptor_cache), "wb", True))

        return descriptors
    
    def extract_single(self, img, verbose=True):
        if not isinstance(img, np.ndarray):
            raise TypeError("img must be numpy array")

        if len(img.shape)!=2:
            raise ValueError("get a input img with shape %s, while this function require a input shape like [width, height, channels]" %str(img.shape))

        keypoints, descripor = self.sift_extractor.detectAndCompute(img, None)
        return descripor[:self.nfeatures]

    def __name__(self):
        return "SIFT"
