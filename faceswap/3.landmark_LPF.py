import sys
sys.path.append("./")

import os
import argparse
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--project_name", default="test2")

    args = parser.parse_args()

    lms_npy = np.load(os.path.join('workspace',args.project_name,'info/target_frames_lms.npy'))

    filtered_lms = []
    filtered_lms.append(lms_npy[0])
    filtered_lms.append(lms_npy[1])

    for i in range(2,len(lms_npy)-2):
        filtered_lms.append(np.array((lms_npy[i-2]+lms_npy[i-1]+lms_npy[i]+lms_npy[i+1]+lms_npy[i+2])/5))

    filtered_lms.append(lms_npy[-2])
    filtered_lms.append(lms_npy[-1])

    np.save(os.path.join('workspace', args.project_name,'info/target_filter_lms.npy'),np.array(filtered_lms))