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

from faceswap.packages.parsing.BiSeNet import BiSeNet

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--project_name", default="test2")
    parser.add_argument("--use_MiVOS", action='store_true')

    args = parser.parse_args()  

    # device setting
    device = torch.device(args.gpu)
    torch.cuda.set_device(device)

    # load pretrained model - face parsing
    parsing_model = BiSeNet(n_classes=19).to(device)
    parsing_model.load_state_dict(torch.load('faceswap/ptnn/79999_iter.pth', map_location=device))
    parsing_model.eval()

    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if args.use_MiVOS:
        os.system(f'python faceswap/packages/MiVOS/interactive_gui.py --images workspace/{args.project_name}/SR_swap_face')
        mivos_mask_path = f'./workspace/{args.project_name}/target_face'
        mivos_mask_list = sorted(glob.glob(mivos_mask_path+'/*.*'))

    target_face_path = f'./workspace/{args.project_name}/target_face'
    target_face_list = sorted(glob.glob(target_face_path+'/*.*'))

    swap_face_path = f'./workspace/{args.project_name}/swap_face'
    swap_face_list = sorted(glob.glob(swap_face_path+'/*.*'))

    save_path = f'./workspace/{args.project_name}/edit_mask'

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
    
    # swap 얼굴을 parsing한다. 
    # target 얼굴을 parsing한다.
    # parsing한 mask와 MiVOS mask를 곱한다.
    # swap 얼굴에서 몇가지만 뽑아낸다.(blur/ erode 포함)
    # 뽑아낸 얼굴에서 
    
    for i in tqdm(range(len(swap_face_list)),desc='edit mask..'):
        target_face = Image.open(target_face_list[i])
        swap_face = Image.open(swap_face_list[i])

        target_face_ = test_transform(target_face).unsqueeze(0).to(device).float()
        swap_face_ = test_transform(swap_face).unsqueeze(0).to(device).float()

        with torch.no_grad():
            # Q > target_mask & swap_mask : save?
            target_mask = parsing_model(F.interpolate(target_face_, (512,512))).max(1)[1]
            swap_mask = parsing_model(F.interpolate(swap_face_, (512,512))).max(1)[1]
    
        if args.use_MiVOS:
            mivos_mask = np.array(Image.open(mivos_mask_list[i]))
            target_mask = target_mask * mivos_mask

        fn_mask = np.zeros((512, 512))
        fn_mask += torch.where(target_mask==1, 1, 0).squeeze().cpu().numpy()
        fn_mask += torch.where(target_mask==2, 1, 0).squeeze().cpu().numpy()
        fn_mask += torch.where(target_mask==3, 1, 0).squeeze().cpu().numpy()
        # fn_mask += torch.where(target_mask==4, 1, 0).squeeze().cpu().numpy() # eye_l
        # fn_mask += torch.where(target_mask==5, 1, 0).squeeze().cpu().numpy() # eye _r
        fn_mask += torch.where(target_mask==10, 1, 0).squeeze().cpu().numpy()
        fn_mask += torch.where(target_mask==11, 1, 0).squeeze().cpu().numpy()
        fn_mask += torch.where(target_mask==12, 1, 0).squeeze().cpu().numpy()
        fn_mask += torch.where(target_mask==13, 1, 0).squeeze().cpu().numpy()
        
        fn_mask = cv2.erode(fn_mask.astype(np.float64)*255, k, iterations=5).astype(np.uint8)/255
        fn_mask = cv2.blur(fn_mask.astype(np.float64)*255, (30, 30)).astype(np.uint8)/255
        
        fn_mask = cv2.resize(fn_mask, (1024,1024))*255
        
        cv2.imwrite(f'{save_path}/{str(i).zfill(5)}.png',fn_mask)

        # # fn_mask2 = np.ones_like(fn_mask)
        # # fn_mask2 -= torch.where(target_mask==4, 1, 0).squeeze().cpu().numpy() # eye_l
        # # fn_mask2 -= torch.where(target_mask==5, 1, 0).squeeze().cpu().numpy() # eye _r
        # # fn_mask2 = cv2.erode(fn_mask2.astype(np.float64)*255, k, iterations=5).astype(np.uint8)/255

        # # fn_mask2 = cv2.resize(fn_mask2, (256,256))

        # # fn_mask2 = np.expand_dims(fn_mask2, 2).repeat(3, axis=2)

        # # fn_mask2 = cv2.blur(fn_mask2.astype(np.float64)*255, (5, 5)).astype(np.uint8)/255
        
        # # Ys = fn_mask2*Ys + (1-fn_mask2)*target_face
        # # Ys = Ys.astype(np.float32)
        # # cv2.imwrite(f"result/{dirs}/mask/{str(i).zfill(5)}.jpg",fn_mask2*255)

        # # tone matching
        # fn_mask_t = np.zeros((512, 512))
        # #fn_mask_t += torch.where(Pt==1, 1, 0).squeeze().cpu().numpy() # skin 
        # fn_mask_t += torch.where(Pt==10, 1, 0).squeeze().cpu().numpy() # nose
        
        # fn_mask_y = np.zeros((512, 512))
        # #fn_mask_y += torch.where(Py==1, 1, 0).squeeze().cpu().numpy()
        # fn_mask_y += torch.where(Py==10, 1, 0).squeeze().cpu().numpy()


