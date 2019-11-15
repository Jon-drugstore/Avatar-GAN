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
sigma_m = 2.0
# sigma_c = 1.0 # line width
sigma_c = 1.0
# rho = 0.997   # noise
rho = 0.996
# original tau = 0.8
#tau = 0.936 # threshold
tau = 0.98

ETF_kernel = 5
ETF_iteration = 0
FDoG_iteration = 0


def rotate_flow(grad_field, theta=90):
    # rotate flow
    theta = theta / 180.0 * math.pi
    flow_field = np.zeros(grad_field.shape, dtype=float)

    cos = math.cos
    sin = math.sin
    asarray = np.asarray
    for i in range(grad_field.shape[0]):
        for j in range(grad_field.shape[1]):
            grad = grad_field[i, j]
            rx = grad[0] * cos(theta) - grad[1] * sin(theta)
            ry = grad[1] * cos(theta) + grad[0] * sin(theta)
            flow_field[i, j] = asarray([rx, ry, 0])
    # pdb.set_trace()
    return flow_field


'''def combine_img(rawimg):
    img = cv2.imread(rawimg, CV_LOAD_IMAGE_GRAYSCALE)
    result = np.zeros(img.shape, dtype=float)
    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            h = result[y, x]

            if h == 0:
                img[y, x] = 0
    cv2.GaussianBlur(img, (3, 3), 0)
    return True'''


def binary_thresholding(s_dog):
    h_img = np.zeros(s_dog.shape, dtype=float)
    h_img[s_dog < tau] = 0
    h_img[s_dog >= tau] = 255
    return h_img

'''
def compute_phi(t_cur_x, t_cur_y):
    phi = 1
    if np.dot(t_cur_x, t_cur_y) < 0:
        phi = -1
    return phi


def compute_ws(i, j, r, c, kernel):
    ws = 0
    if cv2.norm(np.asarray([i, j]), np.asarray([r, c])) < kernel:
        ws = 1
    return ws


def compute_wm(gradmag_x, gradmag_y):
    wm = (1 + np.tanh(gradmag_y - gradmag_x)) / 2
    return wm


def compute_wd(vec_x, vec_y):
    wd = abs(np.dot(vec_x, vec_y))
    return wd


def refine_vec(flow_field, i, j, kernel):
    t_cur_x = flow_field[i, j]
    t_new = np.zeros(2, )
    for r in range(i - kernel, i + kernel + 1):
        for c in range(j - kernel, j + kernel + 1):
            if r < 0 or r >= flow_field.shape[0] or c < 0 or c >= flow_field.shape[1]:
                continue

            t_cur_y = flow_field[r, c]
            phi = compute_phi(t_cur_x, t_cur_y)  # angle < 90, otherwise change direction
            w_s = compute_ws(i, j, r, c, kernel)  # spatial weight
            w_m = compute_wm(cv2.norm(gradientMag[i, j]), cv2.norm(gradientMag[r, c]))  # magnitude weight
            w_d = compute_wd(t_cur_x, t_cur_y)
            t_new = np.add(t_new, (phi * w_s * w_m * w_d * t_cur_y))

    refined_vec = cv2.normalize(t_new, None).reshape(2, )
    return refined_vec'''


def initial_etf(rawimg):
    # print('Getting initial flow Field ...')
    # Gradient Vector Field  -> rotate 90 -> Edge Tangent Flow (ETF)
    normalize = cv2.normalize
    asarray = np.asarray
    
    img = rawimg
    img_norm = normalize(img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_64FC1)
    sobel_j = cv2.Sobel(img_norm, cv2.CV_64F, 1, 0, ksize=5)
    sobel_i = cv2.Sobel(img_norm, cv2.CV_64F, 0, 1, ksize=5)

    gradient_mag0 = cv2.magnitude(sobel_j, sobel_i)
    gradient_mag = normalize(gradient_mag0, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    gradient_field = np.zeros(tuple(img.shape[:2]) + tuple([3]), dtype=float)
    # pdb.set_trace()
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            gj = sobel_j[i, j]
            gi = sobel_i[i, j]
            gradient_field[i, j] = normalize(asarray([gi[0], gj[0], 0]), None).reshape(3, )

    flow_field = rotate_flow(gradient_field, 90)
    return flow_field, gradient_mag


'''def refine_etf(flow_field, kernel=5):
    print('Refining flow Field ...')
    # equation(1) : smooth directions
    refined_field = np.zeros(flow_field.shape, dtype=float)
    for i in range(flow_field.shape[0]):
        for j in range(flow_field.shape[1]):
            update_vec = refine_vec(flow_field, i, j, kernel)
            refined_field[i, j] = update_vec
    return refined_field'''


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
    img = img.astype(float) / 255
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

            direction = (-flow_field[i, j][0], flow_field[i, j][1])  # (x,y) rotate 90, (-y, x)
            if direction[0] == 0 and direction[1] == 0:
                continue

            for pix in range(-t + 1, t):
                # image value

                row = i + int((direction[1] * pix) + 0.5)
                col = j + int((direction[0] * pix) + 0.5)
                if col > img_1 - 1 or col < 0 or row > img_0 - 1 or row < 0:
                    continue

                value = img[row, col]

                # kernel weight
                gau_s_weight = gau_s[abs(pix)]
                if abs(pix) >= len(gau_c):
                    gau_c_weight = 0
                else:
                    gau_c_weight = gau_c[abs(pix)]
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
    img = img.astype(float) / 255
    f_dog = np.zeros(img.shape, dtype=float)
    gau_m = gaussian_kernal(sigma_m)
    s = len(gau_m)
    
    #rnd = np.round
    p = 10**5
    tanh = np.tanh
    img_0, img_1 = img.shape[:2]
    for i in range(t_dog.shape[0]):
        for j in range(t_dog.shape[1]):

            gau_m_acc = -gau_m[0] * t_dog[i, j]
            gau_m_weight_acc = -gau_m[0]

            row0 = i
            col0 = j
            for pix in range(0, s):
                #row = rnd(row0).astype(int)
                #col = rnd(col0).astype(int)
                row = int(row0+0.5)
                col = int(col0+0.5)
                if col > img_1 - 1 or col < 0 or row > img_0 - 1 or row < 0:
                    break

                direction = (-flow_field[row, col][0], -flow_field[row, col][1])
                if direction[0] == 0 and direction[1] == 0:
                    break

                value = t_dog[row, col]
                weight = gau_m[abs(pix)]

                gau_m_acc += value * weight
                gau_m_weight_acc += weight

                row0 += direction[0]
                col0 += direction[1]
                
            if gau_m_weight_acc==0:
                f_dog[i, j] = 1
            elif (gau_m_acc / gau_m_weight_acc) > 0:
                f_dog[i, j] = 1
            else:
                f_dog[i, j] = 1 + tanh(gau_m_acc / gau_m_weight_acc)

    f_dog = cv2.normalize(f_dog, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return f_dog


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

    flow_field, gradient_mag = initial_etf(img)
    t_dog = gradient_dog(img[:, :, 0], flow_field)
    s_dog = flow_dog(img[:, :, 0], t_dog, flow_field)
    s_dog_2 = cv2.normalize(s_dog, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    final = binary_thresholding(s_dog_2)
    
    thres = 180
    #final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, np.ones((3,3)))
    final = cv2.GaussianBlur(final,(7,7),0)
    final[final > thres] = 255
    #final[final <= thres] = 0
    
    # noise removal
    #final = cv2.morphologyEx(final, cv2.MORPH_CLOSE, np.ones((3,3)))
    
    # fill hair  
    if shade:  
        fill = np.zeros(final.shape)
        x, y = (gray[70:186,70:186] < 240).nonzero()
        val = np.median(gray[70+x, 70+y]) - 15
        #print(val)
        ret, binary = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY_INV)
        #ret, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((5,5)))
        final[binary > 0] = 0
    
    
    if save_path is not None:
        cv2.imwrite(save_path, final)

    return final
    
if __name__ == "__main__":
    run_fdog("../static/input_avatar.jpg", "../static/input_FDoG.jpg")
