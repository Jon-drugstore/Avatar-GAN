from deap import base, creator, tools
import robo_draw.img2linelist as i2l
#import matplotlib.pyplot as plt
import random, math, operator, numpy as np, time, copy


def splitKey(individual):
    integer = [int(i) for i in individual]
    decimal = list(map(operator.sub, individual, integer))

    return decimal, integer


def decode(individual):
    # recover route from keys in indiv
    decimal, direction = splitKey(individual)
    order = np.argsort(decimal)

    return direction, order


def keyGen():
    return random.random() * 2
    
    
def runAlgo(draw_list, home=(0, 256), heuristics='LK'):
    print("optimizing...")
    tsp = TSP(draw_list, home, heuristics)
    
    best_ind = tsp.genetic_algo()
    final_list = tsp.getDecodedList(best_ind)
    print("optimized.\n")
    
    return final_list


def runtime(image, heuristics, approx_img_h=256.0, home=(0, 256)):
    draw_list = i2l.img2lines(image, approx_img_h)
    tsp = TSP(draw_list, home, heuristics)

    best = tsp.genetic_algo(test=True)
    return best


class TSP:
    def __init__(self, draw_list, home=(0, 256), heuristics='LK'):
        self.draw_list = draw_list
        self.toolbox = base.Toolbox()
        self.home = home
        self.prevfit = 0
        self.thres = 3
        self.add_cost = 30
        self.LKlist = []
        self.heuristics = heuristics

        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMin)

        #### initialize population ###
        # Attribute generator
        self.toolbox.register("attr_float", keyGen)

        # Structure initializers
        # define 'individual' to be a random-key list encoding route
        self.toolbox.register("individual", tools.initRepeat, creator.Individual, self.toolbox.attr_float,
                              len(draw_list))

        # define the population to be a list of individuals
        self.toolbox.register("population", tools.initRepeat, list, self.toolbox.individual)
        ###############################

        ### Operator registration ###
        # register the goal / fitness function
        self.toolbox.register("evaluate", self.evalRoute)

        # register the crossover operator
        # Uniform crossover, chance of gene swap at 60%
        self.toolbox.register("mate", tools.cxUniform, indpb=0.7)

        # register a mutation operators
        self.toolbox.register("mutate_seq", tools.mutShuffleIndexes, indpb=0.05)
        self.toolbox.register("mutate_dir", tools.mutFlipBit, indpb=0.05)
        self.toolbox.register("mutate_noise", tools.mutGaussian, mu=0, sigma=0.001, indpb=1)  # add a bit of noise

        # operator for selecting individuals for breeding the next
        self.toolbox.register("select", tools.selTournament, tournsize=3)
        # self.toolbox.register("select", tools.selRandom)

        self.toolbox.register("localopt", self.localOpt)
        ################################

    
    def getDecodedList(self, individual):
        direction, order = decode(individual)
        draw_list = list(self.draw_list)
        decoded_list = [draw_list[i] if direction[i] else draw_list[i][::-1] for i in order]

        return decoded_list

    # the goal ('fitness') function to be minimized
    def evalRoute(self, individual):
        # recover route from keys in indiv
        direction, order = decode(individual)
        home = self.home
        draw_list = self.draw_list
        sqrt = math.sqrt
        thres = self.thres
        add_cost = self.add_cost

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
        return fitness,

    def evalTest(self, direction, order):
        home = self.home
        draw_list = self.draw_list
        sqrt = math.sqrt
        thres = self.thres
        add_cost = self.add_cost

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
        
    def eval2Opt(self, i, j, direction, order):
        drawlist = self.draw_list
        home = self.home
        thres = self.thres
        add_cost = self.add_cost
        
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

    def evalSwap(self, i, direction, order):
        drawlist = self.draw_list
        home = self.home
        thres = self.thres
        add_cost = self.add_cost
        
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

    def get2Opt(self, i, j, direction, order):
        drawlist = self.draw_list
        home = self.home

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

    def improvePath(self, delta, order, direction, depth, R):
        key_len = len(order)
        # eval2Opt = self.eval2Opt
        get2Opt = self.get2Opt
        sqrt = math.sqrt
        thres = self.thres
        add_cost = self.add_cost
        limit = 2

        breadth = 6

        # continue search by evaluating vertices
        swap_list = []
        for i, j in ((i, j) for i in range(key_len - 1) for j in range(i, key_len)):
            if (i, j) not in R:
                # diff, diff2, diff3 = eval2Opt(i,j,direction,order)
                x1, y1, x2, y2, x3, y3, x4, y4 = get2Opt(i, j, direction, order)
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
            fdelta, forder, fdirection = self.improvePath(d, sorder, sdirection, depth + 1, R)

            # terminate search if promising path is found
            if fdelta < -0.1:
                break

        return fdelta, forder, fdirection
 
    def localOpt(self, indiv):
        # return indiv
        fit = indiv.fitness.values[0]
        decimal, direction = splitKey(indiv)
        key_len = len(indiv)
        sdec = sorted(decimal)
        # get unique decode
        for k in range(1,key_len):
            if sdec[k]==sdec[k-1]:
                sdec[k] += 1e-10*(0.5-random.random())
        sdec.sort()        
        order = np.argsort(decimal)
        LKlist = self.LKlist
        appLK = LKlist.append
           
        evaluate = self.toolbox.evaluate
        eval2Opt = self.eval2Opt
        evalSwap = self.evalSwap

        repeat = True
        while repeat:
            # swap
            repeat = False
            for i in range(key_len):
                diff = evalSwap(i, direction[:], order[:])
                if diff < -0.1:
                    fit += diff
                    current = order[i]
                    if direction[current] == 0:
                        direction[current] = 1
                    else:
                        direction[current] = 0
                    repeat = True

        repeat = True
        while repeat:
            # 2-opt
            repeat = False
            for i, j in ((i, j) for i in range(key_len - 1) for j in range(i + 1, key_len)):
                diff = eval2Opt(i, j, direction[:], order[:])
                if diff < -0.1:
                    fit += diff
                    # rearrange lists to encode improved path
                    sorder = list(order)
                    te = list(order)
                    te2 = list(direction)
                    if i == 0:
                        change = order[0:j + 1][::-1]
                    else:
                        change = order[j:i - 1:-1]
                    sorder[i:j + 1] = change
                    for m in change:
                        if direction[m] == 0:
                            direction[m] = 1
                        else:
                            direction[m] = 0                            
                    order[:] = sorder
                    
                    repeat = True
                        
                    
        # child is better than parent pop thres, LK
        if self.heuristics=='LK' and self.prevfit + 0.1 > fit:
            repeat = True
            while repeat:
                repeat = False
                lk_id = list(map(operator.add, order, [d / 10.0 for d in direction]))
                if lk_id not in LKlist:
                    appLK(lk_id)
                    # LK
                    delta = 0
                    depth = 0
                    R = []
                    fdelta, forder, fdirection = self.improvePath(delta, order, direction, depth, R)
                    # print(fdelta)
                    if fdelta < -0.1:
                        fit += fdelta
                        order[:] = forder
                        direction[:] = fdirection
                        repeat = True
                        # print(fit)

        # rearrange decimal
        decimal = [x for _, x in sorted(zip(order, sdec))]
        indiv[:] = list(map(operator.add, decimal, direction))
        indiv.fitness.values = fit,
        
        #if abs(fit-evaluate(indiv)[0])>1:
        #    print(fit,evaluate(indiv)[0],self.evalTest(direction, order))
            
        return indiv

    def genetic_algo(self, test=False):
        #random.seed(10) ## testing

        tour_len = len(self.draw_list)

        # CXPB  is the probability with which two individuals are crossed
        # MUTPB is the probability for mutating an individual
        CXPB, MUTPB = 0.8, 0.5
        n = 100  # Population size
        r = 3  # Reproduction, no. of individuals cloned directly from prev population
        m = 0  # # migration, no. of random init new individuals
        p = 0.05  # initial prevfit percentile

        # create an initial population of n unique routes
        pop = self.toolbox.population(n)

        print("Start of evolution")
        start = time.time()

        # Evaluate the entire population
        fitnesses = list(map(self.toolbox.evaluate, pop))
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        # perform a local search for indiv in initial population
        pop[:] = list(map(self.toolbox.localopt, pop))

        fits = [ind.fitness.values[0] for ind in pop]
        self.prevfit = sorted(fits)[int(p * n)]
        print("  Evaluated %i individuals" % len(pop))

        # Variable keeping track of the number of generations
        g = 0
        count = 0
        max_count = 0
        best = min(fits)

        #plt_data = []

        # Begin the evolution
        evaluate = self.toolbox.evaluate

        # exit local search algo when solution does not improve over 10 gens (count = 10)
        # restart count if solution improves (count is reset to 0)
        while count < max(10, max_count + 5):
            # Start new generation
            g = g + 1
            print("-- Generation %i --" % g)

            # Select the mating/mutating pool
            offspring = self.toolbox.select(pop, len(pop) - r - m)
            # Clone the selected individuals
            offspring = list(map(self.toolbox.clone, offspring))

            # Apply crossover on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                # cross two individuals with probability CXPB
                if random.random() < CXPB:
                    spl = int(tour_len / 2)

                    if random.random() < 0.5:
                        child1[:] = [2 - x for x in child1]
                    self.toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
            
            # Apply mutation on the offspring
            for mutant in offspring:
                # mutate an individual with probability MUTPB
                if random.random() < MUTPB:
                    decimal, direction = splitKey(mutant)

                    while True:
                        self.toolbox.mutate_dir(direction)
                        self.toolbox.mutate_seq(decimal)
                        new_mut = list(map(operator.add, decimal, direction))
                        if mutant != new_mut:
                            break
                    mutant[:] = new_mut
                    '''
                    self.toolbox.mutate_dir(direction)
                    self.toolbox.mutate_seq(decimal)
                    mutant[:] = list(map(operator.add, decimal, direction))
                    '''
                    del mutant.fitness.values
            
            # migration
            offspring.extend(self.toolbox.population(m))

            # Newly created individuals
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            # Add small amount of noise to each key in the newly created individuals
            #print("prev", invalid_ind[0])
            #list(map(self.toolbox.mutate_noise, invalid_ind))
            #print("new", invalid_ind[0])
            
            # Evaluate the new individuals
            fitnesses = list(map(evaluate, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit
            inv = len(invalid_ind)

            # Local search to improve each newly created individual
            invalid_ind[:] = list(map(self.toolbox.localopt, invalid_ind))

            # print("  Evaluated %i individuals" % inv)

            # Population is entirely replaced by the offspring and reproduced clones
            offspring.extend(list(tools.selBest(pop, r)))
            pop[:] = offspring

            # Gather all the fitnesses in one list and print the stats
            fits = [ind.fitness.values[0] for ind in pop]

            length = len(pop)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("  Min %s" % min(fits))
            # print("  Max %s" % max(fits))
            # print("  Avg %s" % mean)
            # print("  Std %s" % std)
            print(CXPB, MUTPB)

            # Compare minfit to prev generations to check for convergence 
            minfit = min(fits)
            if abs(best - minfit) < 0.1:
                count += 1
                p = min(0.05 + 0.01 * count, 0.10)
            else:
                p = 0.05
                if max_count < count:
                    max_count = count
                count = 0
            self.prevfit = sorted(fits)[min(int(p * n), n - 1)]  ####################
            best = minfit
            #plt_data.append(best)

        print("-- End of (successful) evolution --")

        best_ind = tools.selBest(pop, 1)[0]
        #plt.plot(plt_data)
        #plt.draw()
        print("Best individual: %s" % (self.toolbox.evaluate(best_ind)))

        if test:
            return best
        return best_ind

