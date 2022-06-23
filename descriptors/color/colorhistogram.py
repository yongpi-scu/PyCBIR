import itertools
import os
import pickle
import numpy as np

class ColorHistogram:
    def __init__(self, n_bin=12, cal_type="local", n_slice=3, cache_dir="cache", normalize=True) -> None:
        ''' count img color histogram

          arguments
            input    : a path to a image or a numpy.ndarray
            n_bin    : number of bins for each channel
            type     : 'global' means count the histogram for whole image
                       'region' means count the histogram for regions in images, then concatanate all of them
            n_slice  : work when type equals to 'region', height & width will equally sliced into N slices
            normalize: normalize output histogram

          return
            type == 'global'
              a numpy array with size n_bin ** channel
            type == 'region'
              a numpy array with size n_slice * n_slice * (n_bin ** channel)
        '''
        # configs for histogram
        self.n_bin = n_bin  # histogram bins
        self.n_slice = n_slice  # slice image
        self.cal_type = cal_type  # global or local
        self.normalize = normalize  # distance type
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def _histogram(self, img):
        n_bin = self.n_bin
        n_slice = self.n_slice
        
        height, width, channel = img.shape
        # slice bins equally for each channel
        bins = np.linspace(0, 256, n_bin+1, endpoint=True)

        if self.cal_type == 'global':
            hist = self._count_hist(img, bins, channel)
        elif self.cal_type == 'local':
            hist = np.zeros((n_slice, n_slice, n_bin ** channel))
            h_silce = np.around(np.linspace(
                0, height, n_slice+1, endpoint=True)).astype(int)
            w_slice = np.around(np.linspace(
                0, width, n_slice+1, endpoint=True)).astype(int)

            for hs in range(len(h_silce)-1):
                for ws in range(len(w_slice)-1):
                    img_r = img[h_silce[hs]:h_silce[hs+1], w_slice[ws]:w_slice[ws+1]]  # slice img to regions
                    hist[hs][ws] = self._count_hist(img_r, bins, channel)
        if self.normalize:
            hist /= np.sum(hist)
        return hist.flatten()

    def _count_hist(self, input, bins, channel):
        n_bin = self.n_bin
        img = input.copy()
        bins_idx = {key: idx for idx, key in enumerate(itertools.product(
            np.arange(n_bin), repeat=channel))}  # permutation of bins
        hist = np.zeros(n_bin ** channel)

        # cluster every pixels
        for idx in range(len(bins)-1):
            img[(input >= bins[idx]) & (input < bins[idx+1])] = idx
        # add pixels into bins
        height, width, _ = img.shape
        for h in range(height):
            for w in range(width):
                b_idx = bins_idx[tuple(img[h, w])]
                hist[b_idx] += 1

        return hist

    def extract_batch(self, dataloader, verbose=True):
        if self.cal_type == 'global':
            descriptor_cache = "colorhistogram_cache-{}-n_bin{}".format(
                self.cal_type, self.n_bin)
        elif self.cal_type == 'local':
            descriptor_cache = "colorhistogram_cache-{}-n_bin{}-n_slice{}".format(
                self.cal_type, self.n_bin, self.n_slice)

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

        if len(img.shape) !=3:
            raise ValueError("get a input img with shape %s, while this function require a input shape like [width, height, channels]" %str(img.shape))

        descripor = self._histogram(img)
        return descripor

    def __name__(self):
        return "colorhistogram"
