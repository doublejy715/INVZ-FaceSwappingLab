import sys
sys.path.append("./")

import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
from torchvision import transforms

from faceswap.nets.AEI_Net import AEI_Net


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--project_name", default="test2")

    args = parser.parse_args()  

    # device setting
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # faceshifter
    swap_model = AEI_Net(512).to(device)
    ckpt_path = "./faceswap/ptnn/G_latest.pth"
    swap_model.load_state_dict(torch.load(ckpt_path, map_location=device))
    swap_model.eval()

    source_id = np.load(os.path.join('workspace', args.project_name,'info/source_id.npy'))
    source_id = torch.tensor(source_id).to(device)

    
    target_face_path = os.path.join('workspace',args.project_name, 'target_face')
    target_path_list = sorted(glob.glob(f"{target_face_path}/*.*"))
    

    for target_face_path in tqdm(target_path_list, desc=">>> Generate swap face"):
        idx = int(os.path.split(target_face_path)[-1].split('.')[0])

        target_face = Image.open(target_face_path)
        target_face_ = test_transform(target_face).unsqueeze(0).to(device).float()
        swap_face = swap_model(target_face_, source_id)[0]
        swap_face = swap_face.squeeze().detach().cpu().numpy().transpose([1, 2, 0]) * 127.5 + 127.5
        swap_face_ = cv2.cvtColor(swap_face, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join('workspace', args.project_name,f'swap_face/{str(idx).zfill(5)}.jpg'), swap_face_)

       