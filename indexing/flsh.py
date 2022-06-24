import os
import pickle
import numpy as np
from collections import defaultdict
from utils.distance import distance


class FLSH(object):
    """FLy Locality Sensitive Hashing (FLSH)
    This implementation of FLSH is based on https://github.com/tian-kun/Fly-LSH
    """

    def __init__(self, input_dim, hash_dim, num_hashtables=1, sampling_rate=0.1, expansion=20, cache_dir="cache", verbose="True", seed=22):

        self.hash_dim = hash_dim
        self.input_dim = input_dim
        self.sampling_rate = sampling_rate
        self.expansion = expansion
        
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.num_hashtables = num_hashtables
        self.seed = seed # fix random seed for reproducibility
        
        self.hash_tables = [defaultdict(list) for _ in range(self.num_hashtables)]
        self._init_projection_matrices()

    def _init_projection_matrices(self):
        """generate random projection matrices using uniform distribution
        """
        projection_matrices_cache = "{}_cache-random_projection-id{}-od{}-seed{}".format(self.__name__(), self.input_dim, self.hash_dim, self.seed)
        try:
            self.projection_matrices = pickle.load(open(os.path.join(self.cache_dir, projection_matrices_cache), "rb", True))
            if self.verbose:
                print("Using cache..., config=%s" %(projection_matrices_cache))
        except:
            if self.verbose:
                print("Generating random projection matrices..., config=%s" % (projection_matrices_cache))
                
            np.random.seed(self.seed)
            # Step 2: Random projection
            num_projections = int(self.sampling_rate * self.input_dim)  # number of projections from PNs to KCs
            projection_matrices = []
            for _ in range(self.num_hashtables):
                projection_matrice = np.random.rand(self.input_dim, self.hash_dim*self.expansion)
                yindices = np.arange(projection_matrice.shape[1])[None, :]
                xindices = projection_matrice.argsort(axis=0)[-num_projections:, :]
                projection_matrice = np.zeros_like(projection_matrice, dtype=bool)
                projection_matrice[xindices, yindices] = True # sparse projection vectors
                projection_matrices.append(projection_matrice)
            
            # projection_matrices = [np.random.rand(self.input_dim, self.hash_dim) for _ in range(self.num_hashtables)]
            
            pickle.dump(projection_matrices, open(os.path.join(self.cache_dir, projection_matrices_cache), "wb", True))
            
            self.projection_matrices = projection_matrices        
      
    def _hash(self, feature, projection_matrice):
        
        # Step 1: Divisive normalization
        feature -= np.mean(feature)        
        # Step 3: Hashing by winner-take-all
        all_activations = np.dot(feature, projection_matrice)
        # import pdb;pdb.set_trace()
        xindices = np.arange(feature.shape[0])[:, None]
        yindices = all_activations.argsort(axis=1)[:, -self.hash_dim:]
        hash_code = np.zeros_like(all_activations, dtype=bool)
        hash_code[xindices, yindices] = True # choose topk activations
        return hash_code.astype(int)

    def __string2array(self, string):
        return np.array([int(i) for i in string])
    
    def __array2string(self, array):
        return "".join([str(i) for i in array])
    
    def insert(self, sample):
        for i, table in enumerate(self.hash_tables):
            hash_codes = self._hash(sample["descriptor"], self.projection_matrices[i])
            for hash_code in hash_codes:
                table[self.__array2string(hash_code)].append(sample)

    def indexing(self, query, depth=3, dist_func="l0"):
        
        q_img, q_feat = query['img_name'], query['descriptor']
        results = dict()
        
        for i, table in enumerate(self.hash_tables):
            q_hash = self._hash(q_feat, self.projection_matrices[i])[0]
            
            for s_hash in table.keys():
                dis = distance(q_hash, self.__string2array(s_hash), dist_func=dist_func)
                for sample in table[s_hash]:
                    s_img, s_cls, s_feat = sample['img_name'], sample['cls'], sample['descriptor']
                    if q_img == s_img:
                        continue
                    if s_img in results:
                        if dis<results[s_img]["dis"]:
                            results[s_img]["dis"]=dis
                    else:
                        results[s_img] = {
                        'img_name': s_img,
                        'dis': dis,
                        'cls': s_cls,
                        'descriptor': s_feat
                    }
        # import pdb;pdb.set_trace()
        results = sorted(results.values(), key=lambda x: x['dis'])
        if depth and depth <= len(results):
            results = results[:depth]
            
        return results
    
    def __name__(self):
        return "FLSH"