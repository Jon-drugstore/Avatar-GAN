#import robo_draw.ur5_draw_r45 as i2l

#import logging
#import urx
import time
import traceback
import numpy as np
import math
import pickle

v = 0.3
a = 0.4
v_air = 0.5
a_air = 0.4
approx_img_h = 256.0    # pixels
img_home = (0, 256)     # pixel coordinates of image home
paperdist = 0.015       # distance for marker lift


def drawShape(rob, final_list, img_home, size = 0.25):
    ### Set robot parameters
    rad = 0.001         # blending radius
    thres = 0.0015      # waypoint threshold
    res = 0.005         # distance between waypoints (in m)


    ### Set the start_pose (corresponding to bottom left corner of drawing) to be
    ### directly below the robot's current position
    pose = rob.getl()   # robot's current position  
    pose[2] = -0.1095   # set to drawing surface height
    pose[3] = 2.2       # adjust orientation of marker end effector 
    pose[4] = 2.2
    pose[5] = 0
    start_pose = pose[:]
    pose_ur = pose[:]  

    ### In our example, the drawing board is rotated at angle of 45 degrees from UR robot base frame
    angle = math.pi / 4
    s_a = math.sin(angle)
    c_a = math.cos(angle)

    ### Set reference pose as top left corner of drawing, i.e. (0,0)
    x = img_home[0] / approx_img_h * size
    y = -img_home[1] / approx_img_h * size
    pose[0] -= x * c_a + y * s_a
    pose[1] -= -x * s_a + y * c_a

    ### Calculate corresponding waypoints and send to robot 
    poselist = []
    lift = 1
    for i in range(len(final_list)):
        cnt = final_list[i]

        if lift:
            # move to above first point in cnt
            x = cnt[0][0][0] * size / approx_img_h
            y = -cnt[0][0][1] * size / approx_img_h
            pose_ur[0] = pose[0] + x * c_a + y * s_a
            pose_ur[1] = pose[1] - x * s_a + y * c_a
            pose_ur[2] = pose[2]
            poselist.append(pose_ur[:])
            # touch paper at first point in cnt
            pose_ur[2] -= paperdist
            poselist.append(pose_ur[:])
            

        # loop from first point to last point
        pose_prev = [pose_ur[0], pose_ur[1]]
        for j in range(len(cnt)):
            point = cnt[j]
            x = point[0][0] * size / approx_img_h
            y = -point[0][1] * size / approx_img_h
            pose_ur[0] = pose[0] + x * c_a + y * s_a
            pose_ur[1] = pose[1] - x * s_a + y * c_a

            if (pose_prev[0] - pose_ur[0]) ** 2 + (pose_prev[1] - pose_ur[1]) ** 2 > res ** 2:
                poselist.append(pose_ur[:])
                pose_prev = [pose_ur[0], pose_ur[1]]
            elif j == len(cnt) - 1 or j == 0:
                poselist.append(pose_ur[:])
                pose_prev = [pose_ur[0], pose_ur[1]]

            # Send waypoints to robot when there are 98 points
            if len(poselist) > 97:
                try:
                    rob.movels(poselist, a, v, rad, threshold=thres)
                except:
                    traceback.print_exc()
                poselist = []

        ### End of current contour
        # lift from paper if finished whole drawing
        if i == len(final_list) - 1:
            lift = 1
        # Else set start point of next line, and lift from paper if necessary
        else:
            # Set start point of next contour
            x = final_list[i + 1][0][0][0] * size / approx_img_h
            y = -final_list[i + 1][0][0][1] * size / approx_img_h
            # lift if distance to start of next contour is more than threshold
            if (pose[0] + x * c_a + y * s_a - pose_prev[0]) ** 2 + (pose[1] - x * s_a + y * c_a - pose_prev[1]) ** 2 > 4 * thres ** 2:
                lift = 1
            else:
                lift = 0

        # add lift from paper if necessary
        if lift:
            pose_ur[2] += paperdist
            poselist.append(pose_ur[:])

        # Send waypoints to robot when there are 97 points
        elif len(poselist) > 96:
            try:
                rob.movels(poselist, a, v, rad, threshold=thres)
            except:
                traceback.print_exc()
            poselist = []

    # Home the marker tip to 20cm above start after finishing
    start_pose[2] += 0.20
    poselist.append(start_pose)
    # Send waypoints to robot
    try:
        rob.movels(poselist, a_air, v_air, rad, threshold=thres)
    except:
        traceback.print_exc()


def run_robo_draw(draw_list, rob, drawing_size, opt_algo='rkgaLK'):
    
    try:
        start = time.time()

        ### Set optimization algorithm
        if opt_algo == 'greedy':
            import robo_draw.greedy as ga
            algo_h = 'None'
        elif opt_algo == 'greedy2opt':
            import robo_draw.greedy as ga
            algo_h = '2opt'
        elif opt_algo == 'greedyLK':
            import robo_draw.greedy as ga
            algo_h = 'LK'
        elif opt_algo == 'rkga2opt':
            import robo_draw.RKGA as ga
            algo_h = '2opt'
        elif opt_algo == 'rkgaLK':
            import robo_draw.RKGA as ga
            algo_h = 'LK'
        
        ### Run optimization
        final_list = ga.runAlgo(draw_list, home=img_home, heuristics=algo_h)
        opt_time = time.time() - start
        
        #i2l.showResult(final_list, filename, 0.20, 0.005)
        
        ### Execute drawing with UR5 robot 
        if rob==None:
            return
            
        # Lift marker from paper
        # rob.translate((0,0,paperdist),a,v) 
        
        # draw
        start = time.time()
        while True:
            try:
                drawShape(rob, final_list, img_home, size=drawing_size)
                print('No. of lines: ', len(final_list))
                print("Optimization Time: ", opt_time)
                print("Drawing Time: ", time.time() - start)
                break
            except:
                print("error")
                
                try:
                    rob.close()
                except:
                    pass
                    
                try:    
                    rob = urx.Robot("192.168.1.5")
                    rob.set_tcp((0, 0, 0.335, 0, 0, 0))
                    rob.set_payload(0.5, (0, 0, 0))
                except:
                    pass
        
    except:
        traceback.print_exc()
            

