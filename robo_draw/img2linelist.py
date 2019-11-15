import cv2
import numpy as np
import scipy.ndimage.morphology as m
import time, math, copy
import matplotlib.pyplot as plt


def skeletonize(img):
    h1 = np.array([[0, 0, 0], [0, 1, 0], [1, 1, 1]])
    m1 = np.array([[1, 1, 1], [0, 0, 0], [0, 0, 0]])
    h2 = np.array([[0, 0, 0], [1, 1, 0], [0, 1, 0]])
    m2 = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]])
    hit_list = []
    miss_list = []
    for k in range(4):
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))
    img = img.copy()
    while True:
        last = img
        for hit, miss in zip(hit_list, miss_list):
            hm = m.binary_hit_or_miss(img, hit, miss)
            img = np.logical_and(img, np.logical_not(hm))
        if np.all(img == last):
            break

    img = np.array(img, dtype=np.uint8)
    return img


def prune(img, n):
    h1 = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    m1 = np.array([[1, 1, 1], [1, 0, 1], [0, 0, 0]])
    h2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    m2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]])
    hit_list = []
    miss_list = []
    for k in range(4):
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))
    img = img.copy()
    pru = np.zeros(img.shape)
    for k in range(n):
        for hit, miss in zip(hit_list, miss_list):
            hm = m.binary_hit_or_miss(img, hit, miss)
            pru = np.logical_or(pru, hm)
            img = np.logical_and(img, np.logical_not(hm))

    img = np.array(img, dtype=np.uint8)
    return img, pru


def extendLineEnd(img, pru, n):
    h1 = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    m1 = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]])
    h2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    m2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]])
    h3 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]])
    m3 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
    hit_list = []
    miss_list = []
    for k in range(4):
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        hit_list.append(np.rot90(h3, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))
        miss_list.append(np.rot90(m3, k))
        # print(hit_list)
    img = img.copy()
    px, py = np.nonzero(pru)
    ppoints = list(zip(px, py))
    for k in range(n):
        for hit, miss in zip(hit_list, miss_list):
            hm = m.binary_hit_or_miss(img, hit, miss)
            hx, hy = np.nonzero(hm)
            hpoints = zip(hx, hy)
            for x0, y0 in hpoints:
                dist0 = 1000000000000000
                a = 0

                for index in range(len(ppoints)):
                    x, y = ppoints[index]
                    d0x = x - x0
                    d0y = y - y0
                    d0 = d0x * d0x + d0y * d0y
                    if d0 < dist0:
                        dist0 = d0
                        a = index
                if dist0 < 3:
                    x, y = ppoints.pop(a)
                    img[x, y] = 1

                    # img = np.logical_or(img, hm)

    img = np.array(img, dtype=np.uint8)

    return img


def getLineEnd(img):
    h1 = np.array([[0, 0, 0], [0, 1, 0], [0, 1, 0]])
    m1 = np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]])
    h2 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 1]])
    m2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]])
    hit_list = []
    miss_list = []
    for k in range(4):
        hit_list.append(np.rot90(h1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m1, k))
        miss_list.append(np.rot90(m2, k))

    line_end = np.zeros(img.shape)
    for hit, miss in zip(hit_list, miss_list):
        hm = m.binary_hit_or_miss(img, hit, miss)
        line_end = np.logical_or(line_end, hm)

    return line_end


def countJunctions(img):
    h1 = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]])
    m1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    h2 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    m2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    h3 = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    m3 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    hit_list = []
    miss_list = []
    for k in range(4):
        hit_list.append(np.rot90(h1, k))
        miss_list.append(np.rot90(m1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m2, k))
        hit_list.append(np.rot90(h3, k))
        miss_list.append(np.rot90(m3, k))
    img = img.copy()
    junc = np.zeros(img.shape)
    for k in range(3):
        for hit, miss in zip(hit_list, miss_list):
            hm = m.binary_hit_or_miss(img, hit, miss)
            junc = np.logical_or(junc, hm)
            hm = cv2.dilate(np.array(hm, dtype=np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
            img = np.logical_and(img, np.logical_not(hm))

    x, y = np.nonzero(junc)
    return len(x)


def removeJunctions(img):
    h1 = np.array([[0, 1, 0], [0, 1, 0], [1, 0, 1]])
    m1 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    h2 = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1]])
    m2 = np.array([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    h3 = np.array([[0, 0, 1], [1, 1, 0], [0, 1, 0]])
    m3 = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    hit_list = []
    miss_list = []
    for k in range(4):
        hit_list.append(np.rot90(h1, k))
        miss_list.append(np.rot90(m1, k))
        hit_list.append(np.rot90(h2, k))
        miss_list.append(np.rot90(m2, k))
        hit_list.append(np.rot90(h3, k))
        miss_list.append(np.rot90(m3, k))
    img = img.copy()
    junc = np.zeros(img.shape)
    for k in range(3):
        for hit, miss in zip(hit_list, miss_list):
            hm = m.binary_hit_or_miss(img, hit, miss)
            junc = np.logical_or(junc, hm)
            hm = cv2.dilate(np.array(hm, dtype=np.uint8), cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3)))
            img = np.logical_and(img, np.logical_not(hm))

    img = np.array(img, dtype=np.uint8)
    return img, junc


def addJunc(jdraw_list, junc):
    jy, jx = np.nonzero(junc)
    jpoints = list(zip(jx, jy))
    draw_list = []
    tr_list = []
    for line in jdraw_list:
        if len(line) == 1:
            x0, y0 = line[0][0]
            dist0 = 1000000000000000
            d0 = []
            for i in range(len(jpoints)):
                x, y = jpoints[i]
                d0x = x - x0
                d0y = y - y0
                d0.append(d0x * d0x + d0y * d0y)
            si = np.argsort(d0)
            if d0[si[0]] < 15:
                line = np.append([np.array([jpoints[si[0]]])], line, axis=0)
                tr_list.append(si[0])
            if len(si) > 1:
                if d0[si[1]] < 15:
                    line = np.append(line, [np.array([jpoints[si[1]]])], axis=0)
                    tr_list.append(si[1])
        else:
            x0, y0 = line[0][0]
            x1, y1 = line[-1][0]
            dist0 = 1000000000000000
            dist1 = 1000000000000000
            a0 = 0
            a1 = 0
            for i in range(len(jpoints)):
                x, y = jpoints[i]
                d0x = x - x0
                d0y = y - y0
                d1x = x - x1
                d1y = y - y1
                d0 = d0x * d0x + d0y * d0y
                d1 = d1x * d1x + d1y * d1y
                if d0 < dist0:
                    dist0 = d0
                    a0 = i
                if d1 < dist1:
                    dist1 = d1
                    a1 = i
                    # print(dist0,dist1)
            if dist0 < 15:
                line = np.append([np.array([jpoints[a0]])], line, axis=0)
                tr_list.append(a0)
            if dist1 < 15:
                line = np.append(line, [np.array([jpoints[a1]])], axis=0)
                tr_list.append(a1)
        draw_list.append(line)

    # add remaining juncs as single point
    for i in range(len(jpoints)):
        if i not in tr_list:
            draw_list.append(np.array([[jpoints[i]]]))

    return draw_list


def cnt2line(cnt_list, line_end):
    jy, jx = np.nonzero(line_end)
    jpoints = zip(jx, jy)

    sqrt = math.sqrt
    #cos_th = 0.9994
    #cos_th = 0.99996
    draw_list = []
    #print(cnt_list[188])
    #c=-1
    for cnt in cnt_list:
        #c+=1
        if len(cnt) < 3:
            draw_list.append(cnt)
        elif len(cnt) == 3:
            '''
            x0,y0 = cnt[0][0]
            x1,y1 = cnt[1][0]
            x2,y2 = cnt[2][0]
            x0 -= x1
            x2 -= x1
            y0 -= y1
            y2 -= y1
            dot = x0*x2+y0*y2
            ''' 
            d0 = abs(cnt[0][0][0] - cnt[2][0][0]) + abs(cnt[0][0][1] - cnt[2][0][1])
            #if dot/sqrt((x0*x0+y0*y0)*(x2*x2+y2*y2)) > cos_th or 
            if d0 < 2 or min([abs(cnt[1][0][0] - x) + abs(cnt[1][0][1] - y) for x, y in jpoints]+[10]) == 0:
                draw_list.append(np.array([cnt[0], cnt[1]]))
            else:
                draw_list.append(cnt)
        else:
            start_line = []
            end_line = []
            start = 1
            end = 0
            for i in range(len(cnt)):
                if start:
                    start_line.append(cnt[i])
                elif end:
                    end_line.append(cnt[i])

                if i == len(cnt) - 1:
                    # circle
                    if start:
                        start_line.append(cnt[0])
                    # line
                    else:
                        '''
                        x0,y0 = cnt[0][0]
                        x1,y1 = cnt[i][0]
                        x2,y2 = cnt[i-1][0]
                        x3,y3 = cnt[1][0]
                        x4,y4 = cnt[i-2][0]
                        x0 -= x1
                        x2 -= x1
                        x3 -= x1
                        x4 -= x1
                        y0 -= y1
                        y2 -= y1
                        y3 -= y1
                        y4 -= y1
                        a0 = x0*x0+y0*y0
                        a2 = x2*x2+y2*y2
                        a3 = x3*x3+y3*y3
                        a4 = x4*x4+y4*y4
                        if a0==0:
                            a0 = a3
                            x0 = x3
                            y0 = y3
                        if a2==0:
                            a2 = a4
                            x2 = x4
                            y2 = y4
                        dot = x0*x2+y0*y2
                        dot2 = x3*x4+y3*y4 
                        '''
                        d01 = abs(cnt[0][0][0] - cnt[i - 1][0][0])
                        d02 = abs(cnt[0][0][1] - cnt[i - 1][0][1])
                        d1 = abs(cnt[1][0][0] - cnt[i - 2][0][0]) + abs(cnt[1][0][1] - cnt[i - 2][0][1])
                        d2 = abs(cnt[0][0][0] - cnt[i - 2][0][0]) + abs(cnt[0][0][1] - cnt[i - 2][0][1])
                        d3 = abs(cnt[1][0][0] - cnt[i - 3][0][0]) + abs(cnt[1][0][1] - cnt[i - 3][0][1])
                        d4 = abs(cnt[i][0][0] - cnt[i - 1][0][0]) + abs(cnt[i][0][1] - cnt[i - 1][0][1])
                        #if dot/sqrt(a0*a2) > cos_th or dot2/sqrt(a3*a4) > cos_th:
                        if ((d01 < 2 and d02 < 2) or d01+d02 + d1 < 4) or (d4 < 2 and (d2 < 2 or d2 + d3 < 4)) or min([abs(cnt[i][0][0] - x) + abs(cnt[i][0][1] - y) for x, y in jpoints]+[10]) == 0:
                            end_line.append(cnt[i])
                    draw_list.append(np.array(end_line + start_line))
                else:
                    if i == len(cnt) - 2:
                        a = 0
                    else:
                        a = i + 2
                    # line  
                    '''
                    x0,y0 = cnt[i+1][0]
                    x1,y1 = cnt[i][0]
                    x2,y2 = cnt[i-1][0]
                    x3,y3 = cnt[a][0]
                    x4,y4 = cnt[i-2][0]
                    x0 -= x1
                    x2 -= x1
                    x3 -= x1
                    x4 -= x1
                    y0 -= y1
                    y2 -= y1
                    y3 -= y1
                    y4 -= y1
                    a0 = x0*x0+y0*y0
                    a2 = x2*x2+y2*y2
                    a3 = x3*x3+y3*y3
                    a4 = x4*x4+y4*y4
                    #print(a0,a2,a3,a4)
                    if a0==0:
                        a0 = a3
                        x0 = x3
                        y0 = y3
                    if a2==0:
                        a2 = a4
                        x2 = x4
                        y2 = y4
                    dot = x0*x2+y0*y2
                    dot2 = x3*x4+y3*y4 
                    '''
                    d01 = abs(cnt[i + 1][0][0] - cnt[i - 1][0][0])
                    d02 = abs(cnt[i + 1][0][1] - cnt[i - 1][0][1])
                    d1 = abs(cnt[a][0][0] - cnt[i - 2][0][0]) + abs(cnt[a][0][1] - cnt[i - 2][0][1])
                    d2 = abs(cnt[i + 1][0][0] - cnt[i - 2][0][0]) + abs(cnt[i + 1][0][1] - cnt[i - 2][0][1])
                    d3 = abs(cnt[a][0][0] - cnt[i - 3][0][0]) + abs(cnt[a][0][1] - cnt[i - 3][0][1])
                    d4 = abs(cnt[i][0][0] - cnt[i - 1][0][0]) + abs(cnt[i][0][1] - cnt[i - 1][0][1])
                    #if dot/sqrt(a0*a2) > cos_th or dot2/sqrt(a3*a4) > cos_th:
                    #if c==91: 
                    #    print(d0,d1,d2,d3,start,end)
                    if ((d01 < 2 and d02 < 2) or d01+d02 + d1 < 4) or (d4 < 2 and (d2 < 2 or d2 + d3 < 4)) or min([abs(cnt[i][0][0] - x) + abs(cnt[i][0][1] - y) for x, y in jpoints]+[10]) == 0:
                        if start:
                            start = 0
                        else:
                            end = 1
                            end_line.append(cnt[i])

    return draw_list


def extractImg(image, approx_img_h=256.0):
    if isinstance(image, str):
        img = cv2.imread(image)
        # Convert to greyscale
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_h, img_w = img.shape
    else:
        img = image
        img_h, img_w = img.shape

    scale = approx_img_h / img_h
    dim = (int(scale * img_w), int(scale * img_h))
    img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)

    return img


def img2lines(image, approx_img_h=256.0, show=False, shade=False):
    # Obtaining Skeleton (list of points to trace)
    gray = extractImg(image, approx_img_h)
    # img_h, img_w, img_c = img.shape

    if show:
        # original
        cv2.imshow('dst', gray)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # Cleaning image

    # gray = cv2.GaussianBlur(gray,(5,5),0)
    # remove noise
    gray[gray > 220] = 255

    # sharpen
    # kernel_sharpening = np.array([[-1,-1,-1],[-1, 9,-1],[-1,-1,-1]])
    kernel_sharpening = np.array([[-1, -1, -1, -1, -1],
                                  [-1, 2, 2, 2, -1],
                                  [-1, 2, 8, 2, -1],
                                  [-2, 2, 2, 2, -1],
                                  [-1, -1, -1, -1, -1]]) / 8.0
    # gray = cv2.filter2D(gray, -1, kernel_sharpening)
    gray = cv2.filter2D(gray, -1, kernel_sharpening)

    # Convert to binary
    ret, binary = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    # binary = cv2.morphologyEx(binary,cv2.MORPH_CLOSE, np.ones((3,3)))

    if show:
        # cleaned 
        cv2.imshow('dst', np.invert(binary))
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    ############################## Thinning ################################
    # find filled shapes
    if shade:
        # initialize shading params
        fill = np.zeros(binary.shape)       # fill mask
        stripes = np.ones(binary.shape)     # striped pattern
        h,w = binary.shape
        spacing = 8                         # generate striped pattern with specified spacing
        for i in range(2*max(h,w)//spacing):
            c = i*spacing
            cv2.line(stripes,(0,c),(c,0), 0, 1)
               
    # find small blobs to outline/fill
    print(cv2.__version__)
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours2 = copy.deepcopy(contours)
    hierarchy2 = copy.deepcopy(hierarchy)
    for i in range(len(contours2)):        
        child = hierarchy2[0][i][2]
        if child == -1:
            #print(len(contours2))
            cnt = contours2[i]
            arclen = cv2.arcLength(cnt,True)
            area = cv2.contourArea(cnt)
            if arclen != 0 and area>50:
                if 4*3.14*area/(arclen*arclen)>0.6:
                    cv2.drawContours(binary, [cnt], 0, 0, -1)
                    cv2.drawContours(binary, [cnt], 0, 1, 1)
                    if shade:
                        cv2.drawContours(fill, [cnt], 0, 1, -1)
        
        if show:
            # cleaned 
            cv2.imshow('dst', np.invert(binary))
            if cv2.waitKey(0) & 0xff == 27:    
                cv2.destroyAllWindows()
        
                
    # erode lines to get remaining shaded areas
    binary2 = binary.copy()    
    binary2 = cv2.erode(binary2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))) #5
    binary2 = cv2.dilate(binary2, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))#3
    binary2 = cv2.morphologyEx(binary2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    
    if show:
        # cleaned 
        cv2.imshow('dst', np.invert(binary2))
        if cv2.waitKey(0) & 0xff == 27:    
            cv2.destroyAllWindows()
            
    # draw remaining shaded areas
    contours, hierarchy = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        cnt = contours[i]
        arclen = cv2.arcLength(cnt,True)
        area = cv2.contourArea(cnt)
        #print(area)
        if arclen != 0 and area>40:
            if shade and (4*3.14*area/(arclen*arclen)>0.4 or area > 100):
                cont_mask = True
            elif (not shade) and (4*3.14*area/(arclen*arclen)>0.7 and area < 100):
                cont_mask = True
            else:
                cont_mask = False            
            
            if cont_mask:
                mask = np.zeros(binary.shape)
                cv2.drawContours(binary, [cnt], 0, 1, 1)
                cv2.drawContours(mask, [cnt], 0, 1, -1)  
                mask[binary2 < 1] = 0
                binary[mask > 0] = 0
                if area>50 and shade:
                    fill[mask > 0] = 1
                    #cv2.drawContours(fill, [cnt], 0, 1, -1)
    if show:
        # cleaned 
        cv2.imshow('dst', np.invert(binary))
        if cv2.waitKey(0) & 0xff == 27:    
            cv2.destroyAllWindows()
                        
    if shade:
    #fore: fill-1 stripes-0 binary-1 
        #fill[binary2==0] = 0
        stripes[fill==0 ] = 1   
        binary[stripes==0] = 1

    '''                       
    binary2 = binary.copy()        
    binary2 = cv2.erode(binary2, np.ones((7,7)), 60)
    if show:
        # cleaned 
        cv2.imshow('dst', np.invert(binary2))
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    contours, hierarchy = cv2.findContours(binary2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        child = hierarchy[0][i][2]
        if child == -1:
            cnt = contours[i]
            arclen = cv2.arcLength(cnt,True)
            area = cv2.contourArea(cnt)
            #print(area)
            if arclen != 0 and area>25:
                cv2.drawContours(binary, [cnt], 0, 0, -1)
                cv2.drawContours(binary, [cnt], 0, 1, 1)
    '''
                    
    # Thin image
    skel = skeletonize(binary)
    # skel = cv2.dilate(skel, np.ones((3,3)))
    # skel = skeletonize(skel)
    if show:
        # skel
        cv2.imshow('dst', np.invert(skel * 255))
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # prune
    prune_count = 6
    pskel, pru = prune(skel, 1)
    junc_count = countJunctions(pskel)
    for k in range(5):
        # print(junc_count)
        # temp = pskel.copy()
        # temp, temp_pru = prune(temp,1)
        # new_count = countJunctions(temp)
        pskel, temp_pru = prune(pskel, 1)  #########
        pru = np.logical_or(pru, temp_pru)  #######
        new_count = countJunctions(pskel)
        # print(new_count)
        if junc_count - new_count < 4:
            prune_count = k + 2
            break
        junc_count = new_count
        # pskel = temp
        # pru = np.logical_or(pru, temp_pru)

    # add small details that were removed from pruning back into the image
    contours, hierarchy = cv2.findContours(skel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        arc_len = cv2.arcLength(cnt, False)
        # print(arc_len)
        if arc_len < 15:
            cv2.drawContours(pskel, [cnt], 0, 1, 1)
    if show:
        # pruned
        cv2.imshow('dst', np.invert(pskel * 255))
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    # extend pruned lines         
    pskel = extendLineEnd(pskel, pru, prune_count)  #########
    pskel = np.array(np.logical_and(binary, pskel), dtype=np.uint8)
    if show:
        # final skel
        cv2.imshow('dst', np.invert(pskel * 255))
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
            ############################### Split into segments, trace ################################

    # remove junctions to get indiv line segments and closed loops
    jskel, junc = removeJunctions(pskel)
    line_end = getLineEnd(jskel)

    jdraw_list = []
    # remove repeated contours for closed loops
    while True:
        jcnt, hierarchy = cv2.findContours(jskel, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(jcnt) == 0:
            break
        elif len(jcnt) == 1:
            jdraw_list.append(jcnt[0])
            break
        else:
            jdraw_list.append(jcnt[0])
            cv2.drawContours(jskel, jcnt, 0, 0, 2)

            # print('a',len(jdraw_list))

    # convert contour to line trace   
    jdraw_list = cnt2line(jdraw_list, line_end)

    # print('b',len(jdraw_list))

    # add junctions to lines
    jdraw_list = addJunc(jdraw_list, junc)

    # print('c',len(jdraw_list))

    if show:
        # junc
        img = cv2.imread(image)
        blank = np.ones(img.shape) * 255
        blank[pskel > 0] = [0, 0, 0]
        junc = cv2.dilate(np.array(junc, dtype=np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        blank[junc > 0] = [0, 0, 255]
        cv2.imshow('dst', blank)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()

    ##########################################################################################

    return jdraw_list


def showResult(draw_list, filename, size=0.20, pen_thickness=0.004, approx_img_h=256.0):
    img_name = filename.split("/")[-1].split(".")[0].split("_")[0]
    #print(img_name)
    
    # Draw lines in different colors
    img = extractImg(filename, approx_img_h)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    blank = np.ones(img.shape) * 255
    cnt_th = max(int(approx_img_h / size * pen_thickness), 1)

    n = 30
    i = 0
    count = 0
    '''
    for line in draw_list:
        color = cv2.cvtColor(np.uint8([[[n * i, 255, 255]]]), cv2.COLOR_HSV2BGR)
        color = color[0][0].tolist()
        cv2.circle(blank, (line[0][0][0],line[0][0][1]), 5, color, -1)
        cv2.circle(blank, (line[-1][0][0],line[-1][0][1]), 5, [0,0,0], -1)
        cv2.polylines(blank, [line], 0, color, 2)
        if i == 6:
            i = 0
        else:
            i += 1
        print(count)
        count += 1
        cv2.imshow('dst', blank)
        if cv2.waitKey(0) & 0xff == 27:
            cv2.destroyAllWindows()
    '''
    blank = np.ones(img.shape) * 255
    cv2.polylines(blank, draw_list, 0, [0, 0, 0], 2)
    for line in draw_list:
        if len(line) == 1:
            cv2.circle(blank, (line[0][0][0], line[0][0][1]), 1, [0, 0, 0], -1)

    #cv2.imshow('dst', blank)
    #if cv2.waitKey(0) & 0xff == 27:
    #    cv2.destroyAllWindows()

    cv2.polylines(img, draw_list, 0, [0, 0, 255], 2)
    for line in draw_list:
        if len(line) == 1:
            cv2.circle(img, (line[0][0][0], line[0][0][1]), 1, [0, 0, 255], -1)

    #cv2.imshow('dst', img)
    #if cv2.waitKey(0) & 0xff == 27:
    #    cv2.destroyAllWindows()
        
    plt.figure(figsize=(16,4))
    plt.subplot(1, 4, 1)
    input_adjust = cv2.imread("static/"+img_name+"_bgrm.jpg")
    plt.imshow(cv2.cvtColor(input_adjust, cv2.COLOR_BGR2RGB))
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 2)
    avatar = cv2.imread("static/"+img_name+"_avatar.jpg", 0)
    plt.imshow(avatar, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 3)
    fdog = cv2.imread("static/"+img_name+"_FDoG.jpg", 0)
    plt.imshow(fdog, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])
    plt.subplot(1, 4, 4)
    plt.imshow(blank)
    plt.xticks([]), plt.yticks([])
    plt.show()


if __name__ == "__main__":
    start = time.time()
    approx_img_h = 256.0

    #filename = 'l17.jpg'
    filename = 'input_FDoG.jpg'
    # filename = 'images/face3.jpeg'
    draw_list = img2lines(filename, approx_img_h, show=True, shade=True)
    # print(draw_list[6])
    print("Time Elapsed: ", time.time() - start)
    print('shapes drawn: ', len(draw_list))
    showResult(draw_list, filename, 0.20, 0.003, approx_img_h)
