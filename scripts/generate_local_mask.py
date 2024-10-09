import numpy as np
from PIL import Image
import argparse
import os
from utils import *


def generate(args=None):
    data = read_csv(csv_filename=args.csv_path)
    data_anomaly_mask = create_mask(data, exception_value=99999)
    centers, labeled_mask = centers_and_labeled_mask(data_anomaly_mask, mask_size=args.mask_size)
    generate_img_mask(data, data_anomaly_mask, centers, labeled_mask, output_path=args.output_path, size=args.size)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, default='./EMAG2/EMAG2_V3_values.csv')
    parser.add_argument("--output_path", type=str, default="./EMAG2/local_masks")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mask_size", type=int, default=10)
    args = parser.parse_args()
    
    generate(args=args)
    