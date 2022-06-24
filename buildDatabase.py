from utils.evaluate import evaluate_batch
from utils.data import Database, DataLoader
import numpy as np
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Classification')
    parser.add_argument('--offline_dataset', type=str, default="/root/workspace/CBIR/Github/CBIR/database/", help='random seed for training.')
    parser.add_argument('--eval_dataset', type=str, default="/root/workspace/CBIR/Github/CBIR/database/", help='random seed for training.')
    parser.add_argument('--descriptor', type=str, default="ColorHistogram", help='choose image descriptor.')
    parser.add_argument('--img_mode', type=str, default="RGB", help='choose image descriptor.')
    parser.add_argument('--indexing_method', type=str, default="lsh", help='choose image descriptor.')
    parser.add_argument('--dist_func', type=str, default="l0", help='choose image descriptor.')
    parser.add_argument('--log_dir', type=str, default="logs", help='choose image descriptor.')
    parser.add_argument('--index_dim', type=int, default=32, help='choose image descriptor.')
    parser.add_argument('--eval', default=True, type=bool, help='whether eval.')
    args = parser.parse_args()
    
    dataloader = DataLoader(data_dir=args.offline_dataset,img_mode=args.img_mode)
    offline_database = Database(dataloader, descriptor_method=args.descriptor, indexing_method=args.indexing_method, index_dim=args.index_dim)
    
    if args.eval:
        fname = "descriptor{}-indexing_method{}-dist_func{}-index_dim{}".format(
            args.descriptor, args.indexing_method, args.dist_func, args.index_dim)
        
        logger = open(os.path.join(args.log_dir, '{}.csv'.format(fname)), 'w')
        logger.write("cls,MAP")
        online_dataloader = DataLoader(data_dir=args.eval_dataset, img_mode=args.img_mode)
        # evaluate database
        APs = evaluate_batch(online_dataloader, offline_database, dist_func="l0")
        # APs = evaluate_batch(online_dataloader, offline_database, dist_func="l0", save_dir=os.path.join(args.log_dir, fname))
        cls_MAPs = []
        for cls, cls_APs in APs.items():
            MAP = np.mean(cls_APs)
            print("Class {}, MAP {}".format(cls, MAP))
            cls_MAPs.append(MAP)
            
            logger.write("\n{},{}".format(cls, MAP))
        print("MMAP", np.mean(cls_MAPs))
        
        logger.write("\n{},{}".format("MMAP", np.mean(cls_MAPs)))
        logger.close()
        