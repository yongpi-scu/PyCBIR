import numpy as np

def AP(label, results, sort=True):
    ''' infer a query, return it's ap

      arguments
        label  : query's class
        results: a dict with two keys, see the example below
                 {
                   'dis': <distance between sample & query>,
                   'cls': <sample's class>
                 }
        sort   : sort the results by distance
    '''
    if sort:
        results = sorted(results, key=lambda x: x['dis'])
    precision = []
    hit = 0
    for i, result in enumerate(results):
        if result['cls'] == label:
            hit += 1
            precision.append(hit / (i+1.))
    if hit == 0:
        return 0.
    return np.mean(precision)