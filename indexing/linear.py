import os
import pickle
import numpy as np
from collections import defaultdict
from utils.distance import distance

class Linear(object):
    """Locality Sensitive Hashing (LSH)
    """

    def __init__(self, verbose="True"):

        self.verbose = verbose
        self.table = []

    def insert(self, sample):
        self.table.append(sample)

    def indexing(self, query, depth=3, dist_func="l0"):
        
        q_img, q_feat = query['img_name'], query['descriptor']
        results = []
        for idx, sample in enumerate(self.table):
            s_img, s_cls, s_feat = sample['img_name'], sample['cls'], sample['descriptor']
            if q_img == s_img:
                continue
            results.append({
                'img_name': s_img,
                'dis': distance(q_feat, s_feat, dist_func=dist_func),
                'cls': s_cls
            })
        
        results = sorted(results, key=lambda x: x['dis'])
        if depth and depth <= len(results):
            results = results[:depth]
        return results