"""
Coherent Line Draw
xiuchao.sui@gmail.com
Nov. 13, 2018
"""
import numpy as np
import cv2
import math

# Parameter Settings:
sigma_ratio = 3
# sigma_ratio = 3
stepsize = 1.0
# sigma_m = 3.0 # degree of coherence
sigma_m = 3.0
# sigma_c = 1.0 # line width
sigma_c = 0.5
# rho = 0.997   # noise
rho = 0.997
# original tau = 0.8
#tau = 0.936 # threshold
tau = 0.95

ETF_kernel = 5
ETF_iteration = 0
FDoG_iteration = 0


def rotate_flow(grad_field, theta=90):
    # rotate flow
    theta = theta / 180.0 * math.pi
    flow_field = np.zeros(grad_field.shape, dtype=float)

    cos = math.cos
    sin = math.sin
    array = np.array
    for i in range(grad_field.shape[0]):
        for j in range(grad_field.shape[1]):
            grad = grad_field[i, j]
            rx = grad[0] * cos(theta) - grad[1] * sin(theta)
            ry = grad[1] * cos(theta) + grad[0] * sin(theta)
            flow_field[i, j] = array([rx, ry])
    # pdb.set_trace()
    return flow_field

def binary_thresholding(s_dog):
    h_img = np.zeros(s_dog.shape, dtype=float)
    h_img[s_dog < tau] = 0
    h_img[s_dog >= tau] = 255
    return h_img

def compute_phi(tx_0, tx_1, ty_0, ty_1): 
    if tx_0*ty_0 + tx_1*ty_1 < 0:
        return -1
    else:
        return 1

def compute_ws(i, j, r, c, kernel):
    a = i-r
    b = j-c
    if a*a + b*b < kernel*kernel:
        return 1
    else:
        return 0

def compute_wm(gradmag_x, gradmag_y):
    return (1 + np.tanh(gradmag_y - gradmag_x)) / 2

def compute_wd(tx, ty):
    return abs(np.dot(tx,ty))

def refine_vec(flow_field, i, j, kernel, gradientMag, ff0, ff1):
    
    min_c, max_c = j-kernel, j+kernel+1
    min_r, max_r = i-kernel, i+kernel+1
    if min_c < 0: min_c = 0
    if min_r < 0: min_r = 0
    if max_c > ff1: max_c = ff1
    if max_r > ff0: max_r = ff0
    rr, cc = max_r-min_r, max_c-min_c
    rc = rr*cc
    ty = flow_field[min_r:max_r,min_c:max_c].reshape(rc,2)
    flow_dot = np.dot(ty,flow_field[i, j])
    
    phi = np.ones((rc,1))
    phi[flow_dot<0] = -1
    r = (i-np.arange(min_r, max_r)).reshape(rr,1)
    c = (j-np.arange(min_c, max_c)).reshape(1,cc)
    w_s = np.zeros((rc,1))
    w_s[(r*r+c*c).reshape(rc,1) < kernel*kernel] = 1
    w_m = compute_wm(gradientMag[i, j], gradientMag[min_r:max_r,min_c:max_c]).reshape(rc,1)
    w_d = abs(flow_dot).reshape(rc,1)
    
    refined_vec = cv2.normalize(np.sum(phi * w_s * w_m * w_d * ty, axis=0), None).reshape(2, )
    return refined_vec

def refine_etf(flow_field, gradientMag, kernel=5):
    #print('Refining flow Field ...')
    # equation(1) : smooth directions
    refined_field = np.zeros(flow_field.shape, dtype=float)
    ff0 = flow_field.shape[0]
    ff1 = flow_field.shape[1]
    return np.array([[refine_vec(flow_field, i, j, kernel, gradientMag, ff0, ff1) for j in range(ff1)] for i in range(ff0)])

def initial_etf(rawimg):
    # print('Getting initial flow Field ...')
    # Gradient Vector Field  -> rotate 90 -> Edge Tangent Flow (ETF)
    normalize = cv2.normalize
    array = np.array
    
    img = rawimg
    img_norm = normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64FC1)
    sobel_j = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=5)
    sobel_i = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=5)

    gradient_mag0 = cv2.magnitude(sobel_j, sobel_i)
    gradient_mag = normalize(gradient_mag0, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    gradient_field = np.zeros(tuple(img.shape[:2]) + tuple([2]), dtype=float)
    # pdb.set_trace()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gj = sobel_j[i, j]
            gi = sobel_i[i, j]
            gradient_field[i, j] = normalize(array([gi[0], gj[0]]), None).reshape(2, )

    flow_field = rotate_flow(gradient_field, 90)
    return flow_field, np.linalg.norm(gradient_mag, axis=2)



def gaussian(x, mu, sigma):
    fx = (np.exp(-(x - mu) * (x - mu) / (2 * sigma * sigma))) / np.sqrt(math.pi * 2.0 * sigma * sigma)
    return fx


def gaussian_kernal(sigma):
    threshold = 0.001
    kernel = []
    append = kernel.append
    ix = 0
    while True:
        ix += 1
        if gaussian(ix, 0, sigma) < threshold:
            break
    for k in range(ix):
        append(gaussian(k, 0, sigma))
    return kernel


def gradient_dog(rawimg, flow_field):
    # print('Calculating Dog (-T ~ T) ...')
    if isinstance(rawimg, str):
        img = cv2.imread(rawimg, 0)  # gray scale
    else:
        img = rawimg
    img = img/ 255.0
    t_dog = np.zeros(img.shape, dtype=float)

    sigma_s = sigma_ratio * sigma_c  # 1.6 * 1
    gau_c = gaussian_kernal(sigma_c)
    gau_s = gaussian_kernal(sigma_s)
    t = len(gau_s)

    img_0, img_1 = img.shape[:2]
    for i in range(img_0):
        for j in range(img_1):
            gau_c_acc = 0
            gau_s_acc = 0
            gau_c_weight_acc = 0
            gau_s_weight_acc = 0

            x, y = flow_field[i, j]  # (x,y) rotate 90, (-y, x)
            d0, d1 = -x, y
            if d0 == 0 and d1 == 0:
                continue

            for pix in range(-t + 1, t):
                # image value

                row = i + int((d1 * pix) + 0.5)
                col = j + int((d0 * pix) + 0.5)
                if col > img_1 - 1 or col < 0 or row > img_0 - 1 or row < 0:
                    continue

                value = img[row, col]

                # kernel weight
                abs_pix = abs(pix)
                gau_s_weight = gau_s[abs_pix]
                if abs_pix >= len(gau_c):
                    gau_c_weight = 0
                else:
                    gau_c_weight = gau_c[abs_pix]
                # sum of kernel weight * image value
                gau_c_acc += value * gau_c_weight
                gau_s_acc += value * gau_s_weight

                # sum of kernel weight
                gau_c_weight_acc += gau_c_weight
                gau_s_weight_acc += gau_s_weight

            if gau_c_weight_acc==0:
                v_c = 1
            else:
                v_c = gau_c_acc / gau_c_weight_acc
                
            if gau_s_weight_acc==0:
                v_s = 1
            else:
                v_s = gau_s_acc / gau_s_weight_acc
            t_dog[i, j] = v_c - rho * v_s

    return t_dog


def flow_dog(rawimg, t_dog, flow_field):
    # print('Calculating Dog (-S ~ S) ...')
    if isinstance(rawimg, str):
        img = cv2.imread(rawimg, 0)  # gray scale
    else:
        img = rawimg
    img = img / 255.0
    f_dog = np.zeros(img.shape, dtype=float)
    gau_m = gaussian_kernal(sigma_m)
    gau_m_0 = gau_m[0] 
    
    nparray = np.array
    npdot = np.dot
    p = 10**5
    tanh = np.tanh
    img_0, img_1 = img.shape[:2]
    i_max, j_max = t_dog.shape[:2]
    for i in range(i_max):
        for j in range(j_max):

            row0 = i
            col0 = j
            gau_m_acc = -gau_m_0 * t_dog[i, j]
            gau_m_weight_acc = -gau_m_0
            
            for weight in gau_m:
                row = int(row0+0.5)
                col = int(col0+0.5)
                if col > img_1 - 1 or col < 0 or row > img_0 - 1 or row < 0:
                    break

                d0, d1 = -flow_field[row, col]
                if d0 == 0 and d1 == 0:
                    break

                row0 += d0
                col0 += d1

                gau_m_acc += t_dog[row, col] * weight
                gau_m_weight_acc += weight                
                
            if gau_m_weight_acc==0:
                f_dog[i, j] = 1                 ### results for ij loop
            else:
                gau_ratio = gau_m_acc / gau_m_weight_acc
                if gau_ratio > 0:
                    f_dog[i, j] = 1
                else:
                    f_dog[i, j] = 1 + tanh(gau_ratio)

    f_dog = cv2.normalize(f_dog, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return f_dog

def genCLD(gray, flow_field):
    t_dog = gradient_dog(gray, flow_field)
    s_dog = flow_dog(gray, t_dog, flow_field)
    s_dog_2 = cv2.normalize(s_dog, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    final = binary_thresholding(s_dog_2)
    return final
    
def run_fdog(img, save_path=None, shade=False):
    if isinstance(img, str):
        img = cv2.imread(img, flags=1)

    if len(img.shape) < 3:  # convert to three channel first
        gray = np.array(img*255, dtype=np.uint8)
        img = np.expand_dims(img, axis=-1)
        img = np.tile(img, (1, 1, 3))
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.normalize(gray, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)

    #edges = cv2.Canny(gray,100,150)
    #cv2.imshow('image',edges)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()

    flow_field, gradient_mag = initial_etf(img)
    for i in range(1):
        flow_field = refine_etf(flow_field, gradient_mag)
    gray_copy = gray.copy()
    cld = genCLD(gray_copy, flow_field)
    for i in range(0):
        gray_copy[cld < 1] = 0
        gray_copy = cv2.GaussianBlur(gray_copy,(3,3),0)
        cld = genCLD(gray_copy, flow_field)
        
    '''    
    cv2.imshow('image',cld)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    cv2.imshow('image',gray_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    #'''
    thres = 200
    #cld = cv2.morphologyEx(cld, cv2.MORPH_CLOSE, np.ones((3,3)))
    cld = cv2.GaussianBlur(cld,(3,3),0)
    cld[cld > thres] = 255
    #cld[cld <= thres] = 0
    #'''
    
    # fill hair  
    if shade:  
        fill = np.zeros(cld.shape)
        x, y = (gray[70:186,70:186] < 240).nonzero()
        val = np.median(gray[70+x, 70+y]) - 15
        #print(val)
        ret, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY_INV)
        #ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5,5)))
        cld[binary > 0] = 0
    
    
    if save_path is not None:
        cv2.imwrite(save_path, cld)

    return cld
    
if __name__ == "__main__":
    run_fdog("../static/gray1_avatar.jpg", "../static/gray1_FDoG.jpg")
