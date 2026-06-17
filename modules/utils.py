# utils.py


import os
import glob


def clear_frames_folder(folder="frames"):
    if not os.path.exists(folder):
        os.makedirs(folder)
        return
    for file in glob.glob(os.path.join(folder, "*.png")):
        os.remove(file)


def initialize_folders(save_frames=False, save_tactics=True):
    os.makedirs("res", exist_ok=True)
    i = 0
    while os.path.exists("res/res" + str(i)):
        i += 1
    res_loc = "res/res" + str(i)
    os.makedirs(res_loc, exist_ok=True)
    #!CLAUDE frames/ · frames_tactics/ 는 해당 옵션이 켜진 경우에만 생성(빈 폴더 방지).
    # os.makedirs("res/res" + str(i) + "/frames", exist_ok=True)
    # os.makedirs("res/res" + str(i) + "/frames_tactics", exist_ok=True)
    if save_frames:
        os.makedirs(res_loc + "/frames", exist_ok=True)
    if save_tactics:
        os.makedirs(res_loc + "/frames_tactics", exist_ok=True)
    return res_loc
