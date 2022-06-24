import numpy as np
from utils.distance import distance
from utils.metrics import AP
from PIL import Image
import os

def evaluate_batch(online_dataloader, offline_database, depth=3, dist_func='l1',verbose=True,save_dir=None):
    ''' infer the whole database

      arguments
        db     : an instance of class Database
        f_class: a class that generate features, needs to implement make_samples method
        depth  : retrieved depth during inference, the default depth is equal to database size
        d_type : distance type
    '''
    classes = online_dataloader.get_class()
    ret = {c: [] for c in classes}

    for idx, sample in enumerate(online_dataloader):
        if verbose:
            print("quering: %d/%d, %s"%(idx, len(online_dataloader), sample["img_name"]))
        results = offline_database.query(sample, depth=depth, dist_func=dist_func)
        ap = AP(sample['cls'], results, sort=False)
        ret[sample['cls']].append(ap)
        
        if save_dir:
            # save results imgs
            os.makedirs(save_dir, exist_ok=True)
            imgs = [np.array(Image.open(sample["img_name"]))]
            for result in results:
                imgs.append(np.array(Image.open(result["img_name"])))
            Image.fromarray(np.hstack(imgs)).save("%s/%s"%(save_dir, sample["img_name"].replace("/","@")))
    return ret
