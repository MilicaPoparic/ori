# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from util import nearestPoint
from game import Directions, Actions
import game

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'OffensiveAgent', second = 'DefensiveAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class OffensiveAgent(CaptureAgent):

    def __init__(self, index):
      self.index = index
      self.observationHistory = []

    def registerInitialState(self, gameState):
      self.start = gameState.getAgentPosition(self.index)
      CaptureAgent.registerInitialState(self, gameState)

    def chooseAction(self, gameState):
      """
      Picks among the actions with the highest Q(s,a).
      """
      def maxLevel(gameState, depth, nextAction):
        currDepth = depth + 1
        if gameState.isOver() or currDepth == 2:  # Terminal Test
          evaluate = self.evaluate(gameState, nextAction,self.index)
          return evaluate
        maxvalue = -999999
        actions = gameState.getLegalActions(self.index)
        for action in actions:
          successor = gameState.generateSuccessor(self.index, action)
          if(self.index % 2 == 0): #paran je
            enemyIdx = 1
            maxvalue = max(maxvalue, minLevel(successor, currDepth, enemyIdx, action))
          else:
            enemyIdx = 0
            maxvalue = max(maxvalue, minLevel(successor, currDepth, enemyIdx, action))
        return maxvalue

      def minLevel(gameState, depth, agentIndex, nextAction):
        minvalue = 999999
        if gameState.isOver()  or depth == 2:  # Terminal Test
          evaluate = self.evaluate(gameState, nextAction, agentIndex)
          return evaluate

        actions = gameState.getLegalActions(agentIndex) # akcije nasih neprijatelja
        for action in actions:
          try:
            if (str(action) == "Stop"):
              continue
            successor = gameState.generateSuccessor(agentIndex, action)
          except:
            continue
          if agentIndex == 2 or agentIndex == 3:
            currDepth = depth + 1
            minvalue = min(minvalue, maxLevel(successor, currDepth, action))
          else:
            agentIndex += 2
            minvalue = min(minvalue, minLevel(successor, depth, agentIndex, action))
        return minvalue

      actions = gameState.getLegalActions(self.index)
      currentScore = -999999
      returnAction = ''
      for action in actions:
        if(str(action) == "Stop"):
            continue
        nextState = gameState.generateSuccessor(self.index, action)
        if (self.index % 2 == 0):  # paran je
          enemyIdx = 1
        else:
          enemyIdx = 0
        score = minLevel(nextState, 0, enemyIdx, action)
        if score > currentScore:
          returnAction = action
          currentScore = score

      return returnAction

    def getSuccessor(self, gameState, action):
      """
      Finds the next successor which is a grid position (location tuple).
      """
      successor = gameState.generateSuccessor(self.index, action)
      pos = successor.getAgentState(self.index).getPosition()
      if pos != nearestPoint(pos):
        # Only half a grid position was covered
        return successor.generateSuccessor(self.index, action)
      else:
        return successor

    def evaluate(self, gameState, action, agentIndex):
      """
      Computes a linear combination of features and feature weights
      """
      features = self.getFeatures(gameState, action, agentIndex)
      weights = self.getWeights(gameState, action)
      return features * weights

    def getFeatures(self, gameState, action, agentIndex):
      tempidx = self.index
      self.index = agentIndex
      features = util.Counter()
      food1 = self.getFood(gameState)
      capsules = gameState.getCapsules()
      foodList = food1.asList()
      walls = gameState.getWalls()
      posX, posY = gameState.getAgentState(self.index).getPosition()
      newx = int(posX)  # nova pozicija x
      newy = int(posY)


      enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
      invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]  # napadaju nas duhovi, mi pacman
      defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]  # mi duhovi njih napadamo (pacmane)


      myState = gameState.getAgentState(self.index)
      myPos = posX, posY  # neke prosledjene koordinate

      # kad smo presli preko i bezimo od neprijatelja
      if (myState.isPacman):
        if myState.numCarrying >= 2: # vracamo se kuci
          newFood = self.getFoodYouAreDefending(gameState).asList()
          minDist = [self.getMazeDistance(myPos, a) for a in newFood]
          features['backToSafeZone'] = min(minDist)
          features['normalGhosts'] = 1  # NE JEDI

        else:  # bezimo od duhova
          features["normalGhosts"] = 0
          for ghost in invaders:
            ghostpos = ghost.getPosition()
            neighbors = Actions.getLegalNeighbors(ghostpos, walls)
            if (newx, newy) == ghostpos and ghost.scaredTimer == 0: #duh nije uplasen, a poklapaju nam se pozicije >> bezimo
              print('BEZIM OD DUHA')
              features["normalGhosts"] = 1
              features['run'] = 100
            elif (newx, newy) == ghostpos and ghost.scaredTimer > 0: #duh uplasen mi ipak idemo na hranu
              features["normalGhosts"] = 0
              #features["eatFood"] = 500
            elif ((newx, newy) in neighbors) and (ghost.scaredTimer == 0): #duh u komsiluku >> bezimo
              features["normalGhosts"] = 1
              dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
              features['ghostInvaders'] = min(dists)
              features['run'] = 100
            elif ((newx, newy) not in neighbors) and (ghost.scaredTimer == 0): #duh nije u komsiluku
              features["normalGhosts"] = 0
            elif ((newx, newy) in neighbors) and ghost.scaredTimer > 0: #u komsiluku uplasen
              features["normalGhosts"] = 1  # idemo na hranu
              #features["eatFood"] = 100

      # MI SMO DUH
      if not myState.isPacman:
        if (len(defenders) == 0): #nema pacmana, idemo na hranu
          features["normalGhosts"] = 0
          features["eatFood"] = 1
        for pacman in defenders:
          pacmanpos = pacman.getPosition()
          neighbors = Actions.getLegalNeighbors(pacmanpos, walls)
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
          features['invaderDistance'] = min(dists) #jurimo najblizeg neprijatelja
          if (newx, newy) == pacmanpos and myState.scaredTimer == 0:  # poklapaju nam se pozicije, pojescemo pacmana
            features["eatFood"] = -5
            features["normalGhosts"] = 1
          elif (newx, newy) == pacmanpos and myState.scaredTimer > 0:  # idemo prema hrani da bismo pobegli od pacmana
            features["normalGhosts"] = 0
            features["scaredGhost"] = 1
          elif ((newx, newy) in neighbors) and (myState.scaredTimer == 0):
            features["normalGhosts"] = 1  # ne idemo ka hrani, nego jurimo pacmana
            features['invaderDistance'] = min(dists)
          elif ((newx, newy) in neighbors) and myState.scaredTimer > 0: #pacman tu mi uplaseni, idemo ka hrani da bi pobegli
            features["normalGhosts"] = 0
            features["scaredGhost"] = 1
          else: #ako nema nista idemo ka hrani
            features["normalGhosts"] = 0

      #idemo ka najblizoj hrani
      if not features["normalGhosts"]:
        if food1[newx][newy]:
          features["eatFood"] = -100
        if len(foodList) > 0:
          tempFood = []
          for food in foodList:
            food_x, food_y = food
            adjustedindex = self.index - self.index % 2
            check1 = food_y > (adjustedindex / 2) * walls.height / 3
            check2 = food_y < ((adjustedindex / 2) + 1) * walls.height / 3 
            if (check1 and check2):
              tempFood.append(food)
          if len(tempFood) == 0:
            tempFood = foodList
          mazedist = [self.getMazeDistance((newx, newy), food) for food in tempFood]
          if min(mazedist) is not None:
            walldimensions = walls.width * walls.height
            features["distanceToFood"] = float(min(mazedist)) / walldimensions

      self.index = tempidx

      return features


    def getWeights(self, gameState, action):
      return {'normalGhosts': 0, 'distanceToFood': -1, 'eatFood': -1, 'invaderDistance': -10,
          'ghostInvaders': -10, 'scaredGhost': -10, 'backToSafeZone': -1, 'run': -1}



class DefensiveAgent(CaptureAgent):

  def __init__(self, index):
    self.index = index
    self.observationHistory = []
    self.idemKuci = False

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """

    def maxLevel(gameState, depth, nextAction):
      currDepth = depth + 1
      if gameState.isOver() or currDepth == 2:  # Terminal Test
        evaluate = self.evaluate(gameState, nextAction, self.index)
        return evaluate
      maxvalue = -999999
      actions = gameState.getLegalActions(self.index)
      for action in actions:
        successor = gameState.generateSuccessor(self.index, action)
        if (self.index % 2 == 0):  # paran je
          enemyIdx = 1
          maxvalue = max(maxvalue, minLevel(successor, currDepth, enemyIdx, action))
        else:
          enemyIdx = 0
          maxvalue = max(maxvalue, minLevel(successor, currDepth, enemyIdx, action))
      return maxvalue


    def minLevel(gameState, depth, agentIndex, nextAction):

      minvalue = 999999
      if gameState.isOver() or depth == 2:  # Terminal Test
        evaluate = self.evaluate(gameState, nextAction, agentIndex)
        return evaluate

      actions = gameState.getLegalActions(agentIndex)  # akcije nasih neprijatelja
      for action in actions:
        try:
          if (str(action) == "Stop"):
            continue
          successor = gameState.generateSuccessor(agentIndex, action)
        except:
          continue
        if agentIndex == 2 or agentIndex == 3:
          currDepth = depth + 1
          minvalue = min(minvalue, maxLevel(successor, currDepth, action))
        else:
          agentIndex += 2
          minvalue = min(minvalue, minLevel(successor, depth, agentIndex, action))
      return minvalue

    actions = gameState.getLegalActions(self.index)
    currentScore = -999999
    returnAction = ''
    for action in actions:
      if (str(action) == "Stop"):
        continue
      nextState = gameState.generateSuccessor(self.index, action)
      if (self.index % 2 == 0):  # paran je
        enemyIdx = 1
      else:
        enemyIdx = 0
      score = minLevel(nextState, 0, enemyIdx, action)

      if score > currentScore:
        returnAction = action
        currentScore = score

    return returnAction

  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action, agentIndex):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action, agentIndex)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action, agentIndex):
    features = util.Counter()
    tempidx = self.index
    self.index = agentIndex
    features = util.Counter()
    food1 = self.getFood(gameState)
    capsules = gameState.getCapsules()
    foodList = food1.asList()
    walls = gameState.getWalls()
    posX, posY = gameState.getAgentState(self.index).getPosition()
    newx = int(posX)  # nova pozicija x
    newy = int(posY)


    food1 = self.getFood(gameState)
    capsules = gameState.getCapsules()
    foodList = food1.asList()

    myState = gameState.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman:
      features['onDefense'] = 0
      if myState.numCarrying >= 2:
        newFood = self.getFoodYouAreDefending(gameState).asList()
        minDist = [self.getMazeDistance(myPos, a) for a in newFood]
        features['distanceToFood'] = min(minDist)
        features['onDefense'] = 1
        if action == Directions.STOP: features['stop'] = 1
        rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
        if action == rev: features['reverse'] = 1
        print(features, action, "SGSAJALKJFBLKAHERKJBNALKFDJNB")
        return features

    enemies = [gameState.getAgentState(i) for i in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)

    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)
      if action == Directions.STOP: features['stop'] = 1
      rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
      if action == rev: features['reverse'] = 1
    else:
      features['onDefense'] = 0
      if food1[newx][newy]:
        features["eatFood"] = -100
      if len(foodList) > 0:
        tempFood = []
        for food in foodList:
          food_x, food_y = food
          adjustedindex = self.index - self.index % 2
          check1 = food_y > (adjustedindex / 2) * walls.height / 3
          check2 = food_y < ((adjustedindex / 2) + 1) * walls.height / 3
          if (check1 and check2):
            tempFood.append(food)
        if len(tempFood) == 0:
          tempFood = foodList
        mazedist = [self.getMazeDistance((newx, newy), food) for food in tempFood]
        if min(mazedist) is not None:
          walldimensions = walls.width * walls.height
          features["distanceToFood"] = float(min(mazedist)) / walldimensions


    self.index = tempidx
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -1, 'reverse': -2, 'distanceToFood': -1}





