import os
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn import preprocessing

class BOW(object):
    """Bag of visual word
    
    """

    def __init__(self, train_samples, k=64, batch_size=1000, cache_dir="cache", verbose="True", seed=22):

        self.k = k
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.seed = seed # fix random seed for reproducibility
        self.train_samples = train_samples
        self.batch_size = batch_size
        
        self._init_kmeans()
        self._init_TFIDF()

    def _init_kmeans(self):
        """generate kmeans
        Using minibatchkmeans for avoiding memory exceeding when facing large-scale datasets.
        """
        kmeans_cache = "{}_cache-kmeans-k{}-seed{}".format(self.__name__(), self.k, self.seed)
        try:
            self.kmeans = pickle.load(open(os.path.join(self.cache_dir, kmeans_cache), "rb", True))
            if self.verbose:
                print("Using cache..., config=%s" %(kmeans_cache))
        except:
            if self.verbose:
                print("Generating KMeans model..., config=%s" % (kmeans_cache))
            descriptors = np.vstack([sample["descriptor"] for sample in self.train_samples])
            kmeans = MiniBatchKMeans(n_clusters=self.k, batch_size=self.batch_size, random_state=self.seed, verbose=0)
            kmeans.fit(descriptors)
            pickle.dump(kmeans, open(os.path.join(self.cache_dir, kmeans_cache), "wb", True))
            self.kmeans = kmeans     
    
    def _init_TFIDF(self):
        """Generate tf-idf feature for training samples
        """
        kmeans_cache = "{}_cache-kmeans-features-k{}-seed{}".format(self.__name__(), self.k, self.seed)
        try:
            features, IDF = pickle.load(open(os.path.join(self.cache_dir, kmeans_cache), "rb", True))
            if self.verbose:
                print("Using cache..., config=%s" %(kmeans_cache))
        except:
            if self.verbose:
                print("Generating KMeans model..., config=%s" % (kmeans_cache))
                
            labels = [self.kmeans.predict(sample["descriptor"]) for sample in self.train_samples]
            TF = np.array([np.bincount(label, minlength=64) for label in labels])
            IDF = np.log((len(self.train_samples) + 1) / (np.sum((TF > 0), axis=0) + 1))
            features = TF*IDF
            features = preprocessing.normalize(features, axis=1, norm='l2')
            
            pickle.dump([features, IDF], open(os.path.join(self.cache_dir, kmeans_cache), "wb", True))
        
        self.IDF = IDF
        self.features = features
    
    def get_features(self):
        return self.features

    def convert(self, sample):
        label = self.kmeans.predict(sample["descriptor"])
        TF = np.bincount(label, minlength=self.k)
        feature = TF[None,:]*self.IDF
        feature = preprocessing.normalize(feature, axis=1, norm='l2')
        return feature
    
    def __name__(self):
        return "BOW"