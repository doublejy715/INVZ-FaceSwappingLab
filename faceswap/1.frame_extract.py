import os
import sys
sys.path.append("./")
import argparse
import cv2

from faceswap.utils.json import save_info_file


sys.path.append("./")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", default="test2")
    parser.add_argument("--video_name", default="video.mp4")
    parser.add_argument("--save_frame", type=int, default=1)
    args = parser.parse_args()

    ret = 1
    frameCount = 0

    json_file_path = f'./workspace/{args.project_name}/info/Info.json'
    video_path = os.path.join('workspace',args.project_name,'target_video',args.video_name)
    save_frame_path = f"./workspace/{args.project_name}/target_frames/{str(frameCount).zfill(5)}.png"

    videoObj = cv2.VideoCapture(video_path)
    fps = videoObj.get(cv2.CAP_PROP_FPS)

    while ret:
        frameId = int(round(videoObj.get(1))) #current frame number
        ret, frame = videoObj.read()
        if frameId % args.save_frame == 0:
            try:
                # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                h, w, _ = frame.shape
                cv2.imwrite(save_frame_path, frame)
                frameCount += 1
            except:
                continue

    Info = {
        "project_name" : args.project_name,
        "fps" : fps,
        "height" : h, 
        "width" : w
    }

    save_info_file(json_file_path,Info)


