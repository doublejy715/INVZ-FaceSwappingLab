import sys
sys.path.append("./")

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--project_name", default="test2")

    args = parser.parse_args()  

    trans_invs = np.load(os.path.join('workspace', args.project_name,'info/trans_invs.npy'))

    save_path = f"./workspace/{args.project_name}/result_frames/"

    target_frame_path = f"./workspace/{args.project_name}/target_frames/*.*"
    target_frame_list = sorted(glob.glob(target_frame_path))

    swap_faces_path = f"./workspace/{args.project_name}/SR_swap_face/*.*"
    swap_faces_list = sorted(glob.glob(swap_faces_path))

    faces_mask_path = f"./workspace/{args.project_name}/edit_mask/*.*"
    faces_mask_list = sorted(glob.glob(faces_mask_path))

    for idx in tqdm(range(len(target_frame_list)), desc=">>> Remerging to original frames"):
        ori_target_frame = np.array(Image.open(target_frame_list[idx]).convert("RGB"))
        swap_face = np.array(Image.open(swap_faces_list[idx]).convert("RGB"))
        
        edit_mask = np.array(Image.open(faces_mask_list[idx]))/255

        edit_mask_ = np.expand_dims(edit_mask, 2).repeat(3, axis=2)
        mask_full = cv2.warpAffine(edit_mask_, trans_invs[idx], (np.size(np.array(ori_target_frame), 1), np.size(np.array(ori_target_frame), 0)), borderValue=(0, 0, 0))
        swap_face_full = cv2.warpAffine(swap_face, trans_invs[idx], (np.size(np.array(ori_target_frame), 1), np.size(np.array(ori_target_frame), 0)), borderValue=(0, 0, 0))

        frame_full = mask_full*swap_face_full + (1-mask_full)*ori_target_frame

        cv2.imwrite(save_path+f'{str(idx).zfill(5)}.png', frame_full[:, :, ::-1])
