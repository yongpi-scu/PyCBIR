import numpy as np
from utils.distance import distance
from utils.metrics import AP

def search_single(query, database, depth=1, dist_func='d1'):
    ''' infer a query, return it's ap

      arguments
        query       : a dict with three keys, see the template
                      {
                        'img': <path_to_img>,
                        'cls': <img class>,
                        'hist' <img histogram>
                      }
        samples     : a list of {
                                  'img': <path_to_img>,
                                  'cls': <img class>,
                                  'hist' <img histogram>
                                }
        db          : an instance of class Database
        sample_db_fn: a function making samples, should be given if Database != None
        depth       : retrieved depth during inference, the default depth is equal to database size
        d_type      : distance type
    '''

    q_img, q_cls, q_feat = query['img_name'], query['cls'], query['descriptor']
    results = []
    for idx, sample in enumerate(database):
        s_img, s_cls, s_feat = sample['img_name'], sample['cls'], sample['descriptor']
        if q_img == s_img:
            continue
        results.append({
            'img': s_img,
            'dis': distance(q_feat, s_feat, dist_func=dist_func),
            'cls': s_cls
        })
    # import pdb;pdb.set_trace()
    results = sorted(results, key=lambda x: x['dis'])
    if depth and depth <= len(results):
        results = results[:depth]
    ap = AP(q_cls, results, sort=False)
    return ap, results


def evaluate_batch(online_database, offline_database, depth=3, dist_func='l1',verbose=True):
    ''' infer the whole database

      arguments
        db     : an instance of class Database
        f_class: a class that generate features, needs to implement make_samples method
        depth  : retrieved depth during inference, the default depth is equal to database size
        d_type : distance type
    '''
    classes = online_database.get_class()
    ret = {c: [] for c in classes}

    for idx, query in enumerate(online_database):
        if verbose:
            print("quering: %d/%d, %s"%(idx, len(online_database), query["img_name"]))
        ap, _ = search_single(query, database=offline_database, depth=depth, dist_func=dist_func)
        ret[query['cls']].append(ap)

    return ret
