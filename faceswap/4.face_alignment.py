import sys
sys.path.append("./")

import os
import cv2
import tqdm
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from faceswap.nets.arcface import Backbone
from faceswap.utils.align_trans import warp_and_crop_face, get_reference_facial_points



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--project_name", default="test2")
    parser.add_argument("--source_name", default="source.png")

    args = parser.parse_args()  

    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # load pretrained model - arcface
    arcface = Backbone(50, 0.6, 'ir_se').to(device)
    arcface.load_state_dict(torch.load('faceswap/ptnn/model_ir_se50.pth', map_location=device))
    arcface.eval()


    source_img_path = os.path.join('workspace', args.project_name, 'source_img', args.source_name)
    target_path = os.path.join('workspace',args.project_name, 'target_frames')
    target_path_list = sorted(glob.glob(f"{target_path}/*.*"))

    source_lm = np.load(os.path.join('workspace', args.project_name,'info/source_lm.npy'))
    target_filter_lms = np.load(os.path.join('workspace', args.project_name,'info/target_filter_lms.npy'))

    ref = get_reference_facial_points(default_square=True)
    # get aligned source face
    source_img = Image.open(source_img_path).convert("RGB")
    source_face, _ = warp_and_crop_face(np.array(source_img), np.array(source_lm), reference_pts=ref, crop_size=(1024, 1024))
    source_face_ = transform(Image.fromarray(source_face)).unsqueeze(0).to(device).float()

    with torch.no_grad():
        source_id = arcface(F.interpolate(source_face_[:, :, 19:237, 19:237], (112, 112), mode='bilinear', align_corners=True))
        source_id = source_id.cpu().numpy()


    # get aligned target face by frame
    trans_inv_list = []
    for target_frame_path in tqdm(target_path_list):
        idx = int(os.path.split(target_frame_path)[-1].split('.')[0])
        target_img = Image.open(target_frame_path).convert("RGB")
        target_face, trans_inv = warp_and_crop_face(np.array(target_img), target_filter_lms[idx], reference_pts=ref, crop_size=(1024, 1024))
        target_face_ = cv2.cvtColor(target_face, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join('workspace', args.project_name,f'target_face/{str(idx).zfill(5)}.jpg'), target_face_)
        trans_inv_list.append(trans_inv)

    np.save(os.path.join('workspace', args.project_name,'info/source_id.npy'),source_id)
    np.save(os.path.join('workspace', args.project_name,'info/trans_invs.npy'),trans_inv_list)
