import os
import cv2
import time
import numpy as np
import tensorflow as tf
from utils.configs import DefaultConfig
from models.avatargan2 import AvatarGAN
from models.bg_removal import run_bg_removal, draw_segment
from models.fdog import run_fdog
from utils.utils import resize_image, adjust_image, convert_grayscale
import robo_draw.img2linelist as i2l
from robo_draw.ur5_draw_r45 import run_robo_draw


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def main(filename, shading = False, rob=None, drawing_size=0.25, table_surface_z=-0.1095, opt_algo='greedy'):
    
    ### create result folder
    #save_folder = "results/infer/"
    save_folder = "static/"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    prog_start = time.time()
    
    ### load image
    img_path = filename
    img_name = img_path.split("/")[-1].split(".")[0]
    input_img = cv2.imread(img_path, flags=1)

    ##########################
    #### IMAGE ADJUSTMENT ####
    ##########################
    
    ### Resize, crop
    print("adjusting input image...")
    start = time.time()    
    input_img, top = resize_image(input_img, save_path=save_folder + img_name + "_resize.jpg")
    adjust_time = time.time() - start
    print("adjust image done, time spent: {:.2f} seconds.\n".format(adjust_time), flush=True)

    ### Background removal
    print("removing background...")
    start = time.time()
    bgrm_img, seg_map = run_bg_removal(input_img, save_path=save_folder + img_name + "_bgrm.jpg", model_type="checkpoint/bgrm/mobile_net_model")
    bgrm_time = time.time() - start
    print("remove background done, time spent: {:.2f} seconds.\n".format(bgrm_time), flush=True)

    ### Convert to gray scale
    print("converting to grayscale...")
    start = time.time()
    #bgrm_img = adjust_image(bgrm_img, gamma=1, val_adj=210, sat_adj=60)
    #bgrm_img = draw_segment(bgrm_img, seg_map)
    cv2.imwrite(save_folder + img_name + "_adjust.jpg", bgrm_img)
    gray_image, b_diff = convert_grayscale(bgrm_img, top, med_value=[135,150], save_path=save_folder + img_name + "_gray.jpg", brightness=True)
    gray_time = time.time() - start
    print("convert to gray_scale done, time spent: {:.2f} seconds.\n".format(gray_time), flush=True)
    

    ###############################
    #### STYLE TRANSFER MODULE ####
    ###############################
    
    ### AvatarGAN
    print("generating avatar...")
    start = time.time()
    config = DefaultConfig()
    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    with tf.Session(config=tf_config) as sess:
        model = AvatarGAN(sess, config)
        avatar = model.infer(gray_image, save_path=save_folder + img_name + "_avatar.jpg", bright_diff=b_diff, is_grayscale=True)
    # avatar = avatar.astype(dtype=np.uint8)
    avatargan_time = time.time() - start
    print("avatar generation done, time spent: {:.2f}\n".format(avatargan_time), flush=True)

    ### Contour Generation
    fdog_start = time.time()
    print("generating line image...")
    start = time.time()
    f_dog = run_fdog(avatar, save_path=save_folder + img_name + "_FDoG.jpg", shade=shading)
    f_dog = f_dog.astype(dtype=np.uint8)
    f_dog_time = time.time() - start
    print("FDoG done, time spent: {:.2f}\n".format(f_dog_time), flush=True)
    

    print("Total time for style transfer: {:.2f}".format(adjust_time, bgrm_time, gray_time, avatargan_time, f_dog_time),
          flush=True)


    ################################
    #### ROBOTIC DRAWING MODULE ####
    ################################

    ### Ordered Pixel-Coordinates Extraction
    print("extracting lines...")
    line_start = time.time()
    filename = save_folder + img_name + "_FDoG.jpg"   
    draw_list = i2l.img2lines(filename, shade=shading, show=False) 
    line_end = time.time()    
    i2l.showResult(draw_list, filename, 0.20, 0.005) ###
    
    ### Optimization. If rob is not None, Robotic Drawing will be executed with the UR5 robot.
    run_robo_draw(draw_list, rob, drawing_size, table_surface_z, opt_algo)
    i2l.showResult(draw_list, filename, 0.20, 0.005) ###
    
    
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument("--filename", type=str, default='CFD/gray1.jpg', help="set input image filename")
    parser.add_argument("--robot_drawing", action='store_true', help="set to True to execute drawing on a UR5 robot")
    parser.add_argument("--opt_algo", type=str, default='rkgaLK', help="set optimization algorithm")
    parser.add_argument("--drawing_size", type=float, default=0.25, help="set size of robot drawing output (in m)")
    parser.add_argument("--table_surface_z", type=float, default=-0.1095, help="set position of table surface along robot z-axis")
    parser.add_argument("--shading", action='store_true', help="set shading option for final robot drawing")
    args = parser.parse_args()
    
    print(args.robot_drawing)
    if args.robot_drawing == True:
        import traceback, logging, urx
        
        # Connect to robot and set robot tcp (position of marker tip)
        try:
            print("Connecting to Robot...")
            logging.basicConfig(level=logging.WARN)
            while True:
                try:
                    print("...")
                    rob = urx.Robot("192.168.1.5")
                    rob.set_tcp((0, 0, 0.335, 0, 0, 0))
                    rob.set_payload(0.5, (0, 0, 0))
                    break
                except KeyboardInterrupt:
                    break
                except:
                    try:
                        rob.close()
                    except:
                        pass
        except:
            traceback.print_exc()
    else:
        rob = None
        
    # Run RoboCoDraw system
    main(args.filename, args.shading, rob, args.drawing_size, args.table_surface_z, args.opt_algo)
