import numpy as np


def augment_img(gt, pre, mode=0):
    '''Kai Zhang (github: https://github.com/cszn)
    '''
    if mode == 0:
        return gt, pre
    elif mode == 1:
        return np.flipud(np.rot90(gt)), np.flipud(np.rot90(pre))
    elif mode == 2:
        return np.flipud(gt), np.flipud(pre)
    elif mode == 3:
        return np.rot90(gt, k=3), np.rot90(pre, k=3)
    elif mode == 4:
        return np.flipud(np.rot90(gt, k=2)), np.flipud(np.rot90(pre, k=2))
    elif mode == 5:
        return np.rot90(gt), np.rot90(pre)
    elif mode == 6:
        return np.rot90(gt, k=2), np.rot90(pre, k=2)
    elif mode == 7:
        return np.flipud(np.rot90(gt, k=3)), np.flipud(np.rot90(pre, k=3))
