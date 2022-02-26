import sys
sys.path.append("./")

import os
import cv2
import glob
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# SR
from faceswap.packages.swinIR.network import SwinIR
from faceswap.packages.swinIR.test import do_SR

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--project_name", default="test2")

    args = parser.parse_args()  

SRnet = SwinIR(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6, 6, 6, 6], embed_dim=240,
                        num_heads=[8, 8, 8, 8, 8, 8, 8, 8, 8],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='3conv')
SRnet.load_state_dict(torch.load("faceswap/ptnn/SR_large.pth")['params_ema'], strict=True)
SRnet.cuda().eval()

swap_face_path = os.path.join('workspace',args.project_name, 'swap_face')
sawp_path_list = sorted(glob.glob(f"{swap_face_path}/*.*"))

for sawp_path in tqdm(sawp_path_list, desc=">>> apply SR to swap face"):
    idx = int(os.path.split(sawp_path)[-1].split('.')[0])
    sawp_Face = cv2.imread(sawp_path).astype(np.float32)
    sawp_Face_ = cv2.cvtColor(sawp_Face, cv2.COLOR_BGR2RGB)
    with torch.no_grad():
        SR_swap_face = do_SR(SRnet, transforms.ToTensor()(sawp_Face_/255).unsqueeze(0).cuda(), 4)
        SR_swap_face = (SR_swap_face.permute(0, 2, 3, 1) * 255).clamp(0, 255).squeeze().detach().cpu().numpy()
    
    cv2.imwrite(os.path.join('workspace', args.project_name,f'SR_swap_face/{str(idx).zfill(5)}.jpg'), SR_swap_face[:, :, ::-1])

        