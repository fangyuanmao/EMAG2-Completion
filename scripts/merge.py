import numpy as np
import os
from tqdm import tqdm
from PIL import Image
from utils import *
import argparse

def merge(args=None):
    os.makedirs(args.output_path, exist_ok=True)
    emag2_original = read_csv(csv_filename=args.EMAG2_V3_path)
    emag2_complete = merge_emag2(emag2_original, args)
    np.savetxt(os.path.join(args.output_path, 'emag2_complete.csv'), emag2_complete, delimiter=', ', fmt='%f')
    # paint(emag2_complete, args.output_path, 'emag2_complete', Normalization=[-200,200], resize=None)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--EMAG2_V3_path", type=str, default="/home/fymao/work/EMAG2/EMAG2_V3_values.csv")
    parser.add_argument("--inpainted_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, default="./")
    parser.add_argument("--local", action='store_true')
    parser.add_argument("--size", type=int, default=540)
    args = parser.parse_args()
    
    merge(args=args)