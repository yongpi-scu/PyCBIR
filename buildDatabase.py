from utils.evaluate import evaluate_batch
from utils.data import Database
import numpy as np
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--offline_dataset', type=str, default="/root/workspace/CBIR/Github/CBIR/database/", help='random seed for training.')
    parser.add_argument('--eval_dataset', type=str, default="/root/workspace/CBIR/Github/CBIR/database/", help='random seed for training.')
    parser.add_argument('--descriptor', type=str, default="SIFT", help='choose image descriptor.')
    # parser.add_argument('--descriptor', type=str, default="ColorHistogram", help='choose image descriptor.')
    parser.add_argument('--eval', default=True, type=bool, help='whether eval.')
    args = parser.parse_args()
    
    offline_database = Database(args.offline_dataset, descriptor_method=args.descriptor,img_mode="GRAY")
    if args.eval:
        online_database = Database(args.eval_dataset, descriptor_method=args.descriptor,img_mode="GRAY")
        # evaluate database
        APs = evaluate_batch(online_database, offline_database, dist_func="l1")
        cls_MAPs = []
        for cls, cls_APs in APs.items():
            MAP = np.mean(cls_APs)
            print("Class {}, MAP {}".format(cls, MAP))
            cls_MAPs.append(MAP)
        print("MMAP", np.mean(cls_MAPs))