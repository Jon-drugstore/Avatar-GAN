import math, time, numpy as np, operator
import robo_draw.img2linelist as i2l


# the goal ('fitness') function to be minimized
def evalRoute(draw_list, direction, order, home):
    sqrt = math.sqrt
    thres = 3
    add_cost = 30

    # air distance from home to start of first line
    first = order[0]
    if direction[first]:
        p0 = (home, draw_list[first][0][0])
    else:
        p0 = (home, draw_list[first][-1][0])

    # air distance when drawing
    p = [(draw_list[current][-1][0] if direction[current] else draw_list[current][0][0],
          draw_list[next][0][0] if direction[next] else draw_list[next][-1][0])
         for current, next in zip(order[:-1], order[1:])]

    p.insert(0, p0)

    # air distance from end of last line to home
    last = order[-1]
    if direction[last]:
        p.append((draw_list[last][-1][0], home))
    else:
        p.append((draw_list[last][0][0], home))

    plist = [(a[0] - b[0], a[1] - b[1]) for a, b in p]
    fit_list = [sqrt(x * x + y * y) for x, y in plist]

    # print("Fitlist: ",fit_list)
    # add cost for lifting pen if fit > thres
    add_cost = [add_cost if fit > thres else 0 for fit in fit_list]

    fitness = sum(fit_list) + sum(add_cost)

    # return air distance in route
    return fitness


def eval2Opt(drawlist, i, j, direction, order, home):
    add_cost = 30
    thres = 3

    if i == 0:
        x1, y1 = home
    else:
        prev = order[i - 1]
        if direction[prev]:
            x1, y1 = drawlist[prev][-1][0]
        else:
            x1, y1 = drawlist[prev][0][0]

    c_i = order[i]
    if direction[c_i]:
        x2, y2 = drawlist[c_i][0][0]
    else:
        x2, y2 = drawlist[c_i][-1][0]

    c_j = order[j]
    if direction[c_j]:
        x3, y3 = drawlist[c_j][-1][0]
    else:
        x3, y3 = drawlist[c_j][0][0]

    if j == len(order) - 1:
        x4, y4 = home
    else:
        next = order[j + 1]
        if direction[next]:
            x4, y4 = drawlist[next][0][0]
        else:
            x4, y4 = drawlist[next][-1][0]

    xof = x1 - x2
    yof = y1 - y2
    xob = x3 - x4
    yob = y3 - y4

    xnf = x1 - x3
    ynf = y1 - y3
    xnb = x2 - x4
    ynb = y2 - y4

    sqrt = math.sqrt
    distnf = sqrt(xnf * xnf + ynf * ynf)
    distnb = sqrt(xnb * xnb + ynb * ynb)
    distof = sqrt(xof * xof + yof * yof)
    distob = sqrt(xob * xob + yob * yob)
    if distnf > thres:
        distnf += add_cost
    if distnb > thres:
        distnb += add_cost
    if distof > thres:
        distof += add_cost
    if distob > thres:
        distob += add_cost

    return distnf + distnb - distof - distob


def evalSwap(drawlist, i, direction, order, home):
    add_cost = 30
    thres = 3

    if i == 0:
        x1, y1 = home
    else:
        prev = order[i - 1]
        if direction[prev]:
            x1, y1 = drawlist[prev][-1][0]
        else:
            x1, y1 = drawlist[prev][0][0]

    c_i = order[i]
    if direction[c_i]:
        x2, y2 = drawlist[c_i][0][0]
        x3, y3 = drawlist[c_i][-1][0]
    else:
        x2, y2 = drawlist[c_i][-1][0]
        x3, y3 = drawlist[c_i][0][0]

    if i == len(order) - 1:
        x4, y4 = home
    else:
        next = order[i + 1]
        if direction[next]:
            x4, y4 = drawlist[next][0][0]
        else:
            x4, y4 = drawlist[next][-1][0]

    xof = x1 - x2
    yof = y1 - y2
    xob = x3 - x4
    yob = y3 - y4

    xnf = x1 - x3
    ynf = y1 - y3
    xnb = x2 - x4
    ynb = y2 - y4

    sqrt = math.sqrt
    distnf = sqrt(xnf * xnf + ynf * ynf)
    distnb = sqrt(xnb * xnb + ynb * ynb)
    distof = sqrt(xof * xof + yof * yof)
    distob = sqrt(xob * xob + yob * yob)
    if distnf > thres:
        distnf += add_cost
    if distnb > thres:
        distnb += add_cost
    if distof > thres:
        distof += add_cost
    if distob > thres:
        distob += add_cost

    return distnf + distnb - distof - distob


def get2Opt(drawlist, i, j, direction, order, home):
    if i == 0:
        x1, y1 = home
    else:
        prev = order[i - 1]
        if direction[prev]:
            x1, y1 = drawlist[prev][-1][0]
        else:
            x1, y1 = drawlist[prev][0][0]

    c_i = order[i]
    if direction[c_i]:
        x2, y2 = drawlist[c_i][0][0]
    else:
        x2, y2 = drawlist[c_i][-1][0]

    c_j = order[j]
    if direction[c_j]:
        x3, y3 = drawlist[c_j][-1][0]
    else:
        x3, y3 = drawlist[c_j][0][0]

    if j == len(order) - 1:
        x4, y4 = home
    else:
        next = order[j + 1]
        if direction[next]:
            x4, y4 = drawlist[next][0][0]
        else:
            x4, y4 = drawlist[next][-1][0]

    return x1, y1, x2, y2, x3, y3, x4, y4


def improvePath(draw_list, delta, order, direction, depth, R, home):
    key_len = len(order)
    sqrt = math.sqrt
    thres = 3
    add_cost = 30
    limit = 2

    '''
    if depth > 1:
        breadth = 1   
    else: 
        breadth = 5
    '''
    '''
    if depth == 0:
        breadth = 4
    if depth == 3:
        breadth = 2   
    else: 
        breadth = 3
    '''
    breadth = 6

    # continue search by evaluating vertices
    swap_list = []
    for i, j in ((i, j) for i in range(key_len - 1) for j in range(i, key_len)):
        if (i, j) not in R:
            # diff, diff2, diff3 = eval2Opt(i,j,direction,order)
            x1, y1, x2, y2, x3, y3, x4, y4 = get2Opt(draw_list, i, j, direction, order, home)
            xof = x1 - x2
            yof = y1 - y2
            xnb = x2 - x4
            ynb = y2 - y4

            distnb = sqrt(xnb * xnb + ynb * ynb)
            distof = sqrt(xof * xof + yof * yof)

            if distnb > thres:
                distnb += add_cost
            if distof > thres:
                distof += add_cost

            # promising vertex
            if delta + distnb - distof < -0.1:
                xob = x3 - x4
                yob = y3 - y4

                xnf = x1 - x3
                ynf = y1 - y3

                distnf = sqrt(xnf * xnf + ynf * ynf)
                distob = sqrt(xob * xob + yob * yob)
                if distnf > thres:
                    distnf += add_cost
                if distob > thres:
                    distob += add_cost

                new_delta = delta + distnf + distnb - distof - distob

                # terminate if improvement is found
                if new_delta < -0.1:
                    # rearrange lists to encode improved path
                    sorder = list(order)
                    sdirection = list(direction)
                    if i != j:
                        if i == 0:
                            change = order[0:j + 1][::-1]
                        else:
                            change = order[j:i - 1:-1]
                        sorder[i:j + 1] = change
                        for m in change:
                            if sdirection[m] == 0:
                                sdirection[m] = 1
                            else:
                                sdirection[m] = 0
                    else:
                        current = order[i]
                        if sdirection[current] == 0:
                            sdirection[current] = 1
                        else:
                            sdirection[current] = 0
                    # terminate and return improved path
                    return new_delta, sorder, sdirection

                # store promising vertex if no improvement
                else:
                    swap_list.append((distnb - distob, new_delta, i, j))

    # terminate if depth = limit                
    if depth == limit:
        return delta, order, direction

    if len(swap_list):
        # continue search with promising vertices, sorted by diff3
        slist = [(d, i, j) for _, d, i, j in sorted(swap_list)]
    else:
        # terminate if no promising vertex is found
        return delta, order, direction

    # limit search to promising vertices with top most negative diff3
    if len(slist) > breadth:
        slist = slist[:breadth]

    # search for improved path
    for d, i, j in slist:
        # store prev swaps
        sR = list(R)
        sR.append((i, j))

        # rearrange lists to encode promising paths
        sorder = list(order)
        sdirection = list(direction)
        if i != j:
            # swap order
            if i == 0:
                change = sorder[0:j + 1][::-1]
            else:
                change = sorder[j:i - 1:-1]
            sorder[i:j + 1] = change
            # swap direction
            for m in change:
                if sdirection[m] == 0:
                    sdirection[m] = 1
                else:
                    sdirection[m] = 0
        else:
            # swap direction
            current = order[i]
            if sdirection[current] == 0:
                sdirection[current] = 1
            else:
                sdirection[current] = 0

        # search for improvements to promising path
        fdelta, forder, fdirection = improvePath(draw_list, d, sorder, sdirection, depth + 1, R, home)

        # terminate search if promising path is found
        if fdelta < -0.1:
            break

    return fdelta, forder, fdirection


def localOpt(draw_list, order, direction, fit, home, LK=False):
    key_len = len(order)

    ## swap & 2-opt. Repeat search until no more improvement observed.
    repeat = True
    while repeat:
        repeat = False
        # swap    
        for i in range(key_len):
            diff = evalSwap(draw_list, i, direction, order, home)
            if diff < -0.1:
                fit += diff
                current = order[i]
                if direction[current] == 0:
                    direction[current] = 1
                else:
                    direction[current] = 0
                repeat = True

        # 2-opt
        for i, j in ((i, j) for i in range(key_len - 1) for j in range(i + 1, key_len)):
            diff = eval2Opt(draw_list, i, j, direction, order, home)
            if diff < -0.1:
                fit += diff
                if i == 0:
                    change = order[0:j + 1][::-1]
                else:
                    change = order[j:i - 1:-1]
                order[i:j + 1] = change
                # swap direction for opt segment
                for m in change:
                    if direction[m] == 0:
                        direction[m] = 1
                    else:
                        direction[m] = 0
                repeat = True

    fit_2opt = fit

    ## LK. Repeat search until no more improvement observed.
    if LK:
        repeat = True
        while repeat:
            repeat = False
            
            delta = 0
            depth = 0
            R = []
            fdelta, forder, fdirection = improvePath(draw_list, delta, order, direction, depth, R, home)
            if fdelta < -0.1:
                fit += fdelta
                order[:] = forder
                direction[:] = fdirection
                repeat = True

    return fit_2opt, fit, order, direction


def runAlgo(draw_list, home=(0, 256), test=False, heuristics='None'):
    # parameters used for cost calculation
    add_cost = 30
    thres = 3 * 3
    cx, cy = home
    
    # initialize variables
    n = len(draw_list)
    total_dist = 0
    order = []
    direction = [0] * n
    indexes = []

    sqrt = math.sqrt
    iapp = indexes.append
    oapp = order.append
    for i in range(n):
        ## Selecting i_th line using greedy
        best_ind = 0
        best_dir = 0
        best_dist = 10000000
        for ind in range(n):
            ## if line is not already drawn:
            if ind not in indexes:
                # calculate distance from current pose to start and end of line 
                p0x, p0y = draw_list[ind][0][0]
                p1x, p1y = draw_list[ind][-1][0]

                p0x -= cx
                p0y -= cy
                p1x -= cx
                p1y -= cy
                dist0 = p0x * p0x + p0y * p0y
                dist1 = p1x * p1x + p1y * p1y

                # store if distance is the least seen so far
                if dist0 < best_dist:
                    best_dist = dist0
                    best_ind = ind
                    best_dir = 1

                if dist1 < best_dist:
                    best_dist = dist1
                    best_ind = ind
                    best_dir = 0

        # add line with least distance to the plan
        oapp(best_ind)                  # update order
        direction[best_ind] = best_dir  # update direction

        # update list of lines drawn and total cost
        iapp(best_ind)
        total_dist += sqrt(best_dist)
        if best_dist > thres:
            total_dist += add_cost

        # set endpoint of line as new current pose
        line = draw_list[best_ind]
        if best_dir:
            cx, cy = line[-1][0]
        else:
            cx, cy = line[0][0]
            

    # Add distance from last line to home to total cost
    px = cx - home[0]
    py = cy - home[1]
    distp = px * px + py * py
    total_dist += sqrt(distp)
    if distp > thres:
        total_dist += add_cost

    
    if heuristics=='2opt':  
        # greedy+2opt 
        opt_dist, final_dist, ford, fdir = localOpt(draw_list, order, direction, total_dist, home, LK=False)
        final_list = [draw_list[i] if fdir[i] else draw_list[i][::-1] for i in ford]                
    elif heuristics=='LK':
        # greedy+LK
        opt_dist, final_dist, ford, fdir = localOpt(draw_list, order, direction, total_dist, home, LK=True)
        final_list = [draw_list[i] if fdir[i] else draw_list[i][::-1] for i in ford]
    else:
        # greedy
        final_list = [draw_list[i] if direction[i] else draw_list[i][::-1] for i in order]
        opt_dist = final_dist = total_dist

    
    if test:
        return final_list, total_dist, opt_dist, final_dist
    else:
        print('Path Fitness Cost: ', final_dist)
        return final_list

