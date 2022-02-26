import sys
sys.path.append("./")

import os
import glob
import argparse
import numpy as np
from PIL import Image

from facenet_pytorch import MTCNN
import face_alignment
from faceswap.utils.utils import get_lm_68_to_5

from tqdm import tqdm


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default="test2")
    parser.add_argument("--source_name", default="source.png")
    parser.add_argument("--target_dir", default="images/lily_shadow_2")
    parser.add_argument("--detect_type", default="Face_align", help='there is two types, "MTCNN","Face_align"')

    args = parser.parse_args()

    source_path = os.path.join('workspace', args.project_name, 'source_img', args.source_name)
    target_path = os.path.join('workspace',args.project_name, 'target_frames')

    source_img = Image.open(source_path).convert("RGB")
    target_frames_path = sorted(glob.glob(f'{target_path}/*.*'))
    target_frames_landmark = []

    # MTCNN detect type
    if args.detect_type == "MTCNN":
        mtcnn = MTCNN()

        _, _, source_lms = mtcnn.detect(source_img, landmarks=True)
        source_lms = np.array(source_lms[0])

        # record target frames landmark
        for target_frame_path in tqdm(target_frames_path, desc=">>> detecting face landmark"):
            target_frame = Image.open(target_frame_path).convert("RGB")
            _, _, target_lms = mtcnn.detect(target_frame, landmarks=True)
            target_lms = np.array(target_lms[0])
            target_frames_landmark.append(target_lms)


    # Face_align
    elif args.detect_type == "Face_align":
        fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
        source_preds = fa.get_landmarks(np.array(source_img))
        source_lms = get_lm_68_to_5(np.array(source_preds))
        
        for target_frame_path in tqdm(target_frames_path, desc=">>> detecting face landmark"):
            target_frame = Image.open(target_frame_path).convert("RGB")
            target_preds = fa.get_landmarks(np.array(target_frame))
            target_lms = get_lm_68_to_5(np.array(target_preds))
            target_frames_landmark.append(target_lms)
            
    np.save(os.path.join('workspace', args.project_name,'info/source_lm.npy'),source_lms)
    np.save(os.path.join('workspace', args.project_name,'info/target_frames_lms.npy'),np.array(target_frames_landmark))


