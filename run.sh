#!/bin/bash
mkdir workspace

read -p ">>> Project Name : " project_name
echo ">>> Project Name is '$project_name'"
mkdir workspace/$project_name

echo ">>> Setting your workspace..."
mkdir workspace/$project_name/source_img
mkdir workspace/$project_name/target_video
mkdir workspace/$project_name/target_frames
mkdir workspace/$project_name/target_face
mkdir workspace/$project_name/swap_face
mkdir workspace/$project_name/SR_swap_face
mkdir workspace/$project_name/MiVOS_masks
mkdir workspace/$project_name/edit_mask
mkdir workspace/$project_name/result_frames
mkdir workspace/$project_name/result_video
mkdir workspace/$project_name/info

# echo ">>> start extract frame from video"
# python faceswap/1.frame_extract.py --project_name ${project_name} --video_name video.mp4
# echo ">>> End extract frame from video"

# echo ">>> start face detection"
# python faceswap/1.frame_extract.py --project_name ${project_name} --video_name video.mp4
# echo ">>> End face detection"



# echo ">>> landmark smoothing..."
# python faceswap/3.landamrk_LPF.py --project_name ${project_name}
