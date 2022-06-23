import numpy as np

def distance(v1, v2, dist_func='l0'):
    # import pdb;pdb.set_trace()
    assert v1.shape == v2.shape, "shape of two vectors need to be same!"
    if dist_func == 'l0':
        # hamming distance
        # reference to wikipedia https://en.wikipedia.org/wiki/Hamming_distance
        return np.sum([el1 != el2 for el1, el2 in zip(v1, v2)])
    elif dist_func == 'l1':
        return np.sum(np.absolute(v1 - v2))
    elif dist_func == 'l2':
        return np.sum((v1 - v2) ** 2)
    else:
        raise NotImplementedError("distance func %s not implemented!"%dist_func)

