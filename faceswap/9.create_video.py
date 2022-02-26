import sys
sys.path.append("./")

import os
import glob
import time
import argparse

from moviepy.video.io.ImageSequenceClip import ImageSequenceClip


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--project_name", default="test2")
    parser.add_argument("--fps", type=int, default=24)

    args = parser.parse_args()  

    time1 = time.time()

    eidt_frame_path = f"./workspace/{args.project_name}/result_frames/"
    save_path = f"./workspace/{args.project_name}/result_video/"

    print(">>> Start transfer video to frames")
    path = os.path.join(eidt_frame_path,'*.*')
    image_filenames = sorted(glob.glob(path))

    clips = ImageSequenceClip(image_filenames,fps = args.fps)
    os.makedirs(f'{save_path}',exist_ok=True)
    clips.write_videofile(os.path.join(f'{save_path}','result.mp4'))
    print(">>> End transfer")