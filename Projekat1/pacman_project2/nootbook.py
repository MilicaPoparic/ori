def getFeatures(self, gameState, action, agentIndex):
    tempidx = self.index
    self.index = agentIndex
    features = util.Counter()
    # successor = self.getSuccessor(gameState, action)
    food1 = self.getFood(gameState)
    capsules = gameState.getCapsules()
    foodList = food1.asList()
    walls = gameState.getWalls()
    posX, posY = gameState.getAgentState(self.index).getPosition()
    newx = int(posX)  # nova pozicija x
    newy = int(posY)

    # Get set of invaders and defenders
    enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
    invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]  # napadaju nas duhovi, mi pacman
    defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]  # mi duhovi njih napadamo pacmane

    # myState = successor.getAgentState(self.index)

    myState = gameState.getAgentState(self.index)
    myPos = posX, posY  # neke prosledjene koordinate

    # kad smo presli preko i bezimo od neprijatelja, MORAMO DA GA VRATIMOOOOOOO KUCI
    if (myState.isPacman):  # vracamo se kuci
        if myState.numCarrying >= 5:
            newFood = self.getFoodYouAreDefending(gameState).asList()
            minDist = [self.getMazeDistance(myPos, a) for a in newFood]
            #  features['distanceToFood'] = min(minDist)
            features['backToSafeZone'] = min(minDist)
            features['normalGhosts'] = 1  # NE JEDI

        else:  # bezimo od duhova
            for ghost in invaders:
                ghostpos = ghost.getPosition()
                neighbors = Actions.getLegalNeighbors(ghostpos, walls)
                if (newx, newy) == ghostpos and ghost.scaredTimer == 0:
                    print("pojesce nas normaln duh")
                    features["normalGhosts"] = 1  # pojesce nas normalan duh, ne idemo na hranu nego bezimo
                elif (newx, newy) == ghostpos and ghost.scaredTimer > 0:
                    print("duh uplasen idemo na hranu")
                    features["normalGhosts"] = 0
                    features["eatFood"] = 500
                elif ((newx, newy) in neighbors) and (ghost.scaredTimer == 0):
                    print("komsije, nije uplasen")
                    features["normalGhosts"] = 1
                    dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                    features['ghostInvaders'] = min(dists)
                elif ((newx, newy) in neighbors) and ghost.scaredTimer > 0:
                    print("komsije uplsen")
                    features["normalGhosts"] = 1  # ignorisemo situaciju
                    features["eatFood"] = 0.100
                # else:
                #   dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
                #   if(min(dists) <= 2 and ghost.scaredTimer == 0):  # treba da duh nije uplasen
                #     print("BEZIM")
                #     features['ghostInvaders'] = min(dists)   # ako je blizu ne jedemo
                #     features['normalGhosts'] = 1  # ako je blizu ne jedemo
                #     print(features, "bezimo")
                #   elif(min(dists) > 2 and ghost.scaredTimer == 0):
                #     print("ne bezim")
                #     features['eatFood'] = 100

    # MI SMO DUH
    if not myState.isPacman:
        if (len(defenders) == 0):
            features["normalGhosts"] = 0
            features["eatFood"] = 100
        for pacman in defenders:
            pacmanpos = pacman.getPosition()
            neighbors = Actions.getLegalNeighbors(pacmanpos, walls)
            dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
            features['invaderDistance'] = min(dists)
            if (newx, newy) == pacmanpos and myState.scaredTimer == 0:  # JEDEMO njega PACMANA
                features["eatFood"] = 200
                features["normalGhosts"] = 1
            elif (newx, newy) == pacmanpos and myState.scaredTimer > 0:  # idemo prema hrani da bismo pobegli od pacmana
                features["normalGhosts"] = 0  # bila 0
                features["scaredGhost"] = 1
            elif ((newx, newy) in neighbors) and (myState.scaredTimer == 0):
                print('mi smo duh, jurimo ')
                features["normalGhosts"] = 1  # ignorisemo situaciju
                features[
                    "eatFood"] = 0.100  # ako nam neprijatelj u komsiluku, idmeo da ga POJEDEMO, potencijalno dobar potez
            elif ((newx, newy) in neighbors) and myState.scaredTimer > 0:
                features["normalGhosts"] = 0  # ignorisemo situaciju, bila 0
                features["scaredGhost"] = 1
            else:
                features["normalGhosts"] = 0
    # na pocetku smo duhovi, nismo uplaseni, nemamo hranu blizu, nemamo neprijatelje, cilj je da predjemo preko
    # trazimo najblizu hranu!!!

    if not features["normalGhosts"]:
        if food1[newx][newy]:
            features["eatFood"] = 100.0
        if len(foodList) > 0:
            tempFood = []
            for food in foodList:
                food_x, food_y = food
                adjustedindex = self.index - self.index % 2
                check1 = food_y > (adjustedindex / 2) * walls.height / 3  # ?????????????
                check2 = food_y < ((adjustedindex / 2) + 1) * walls.height / 3  # ??????????
                if (check1 and check2):
                    tempFood.append(food)
            if len(tempFood) == 0:
                tempFood = foodList
            mazedist = [self.getMazeDistance((newx, newy), food) for food in tempFood]
            if min(mazedist) is not None:
                walldimensions = walls.width * walls.height
                # features["distanceToFood"] = float(min(mazedist)) / walldimensions
                features["distanceToFood"] = float(min(mazedist)) / walldimensions
                print(features)
    # treba da ga vratimo kuci
    self.index = tempidx
    return features


def getWeights(self, gameState, action):
    return {'normalGhosts': 0, 'distanceToFood': -1, 'eatFood': -1, 'invaderDistance': -10,
            'ghostInvaders': 0.00195, 'scaredGhost': -1, 'backToSafeZone': 0.00195}
# return {'normalGhosts':-20, 'distanceToFood': -1, 'eatFood': -1, 'invaderDistance': -10, 'run': -10,
#         'ghostInvaders': 0.00195, 'scaredGhost': -20, 'backToSafeZone': -1, 'reverse': 10}
