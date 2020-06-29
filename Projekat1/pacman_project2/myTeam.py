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
      actions = gameState.getLegalActions(self.index)

      # You can profile your evaluation time by uncommenting these lines
      # start = time.time()
      values = [self.evaluate(gameState, a) for a in actions]
      # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

      maxValue = max(values)
      bestActions = [a for a, v in zip(actions, values) if v == maxValue]

      foodLeft = len(self.getFood(gameState).asList())

      return random.choice(bestActions)

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

    def evaluate(self, gameState, action):
      """
      Computes a linear combination of features and feature weights
      """
      features = self.getFeatures(gameState, action)
      weights = self.getWeights(gameState, action)
      return features * weights

    def getFeatures(self, gameState, action):

      features = util.Counter()
      successor = self.getSuccessor(gameState, action)
      food1 = self.getFood(gameState)
      capsules = gameState.getCapsules()
      foodList = food1.asList()
      walls = gameState.getWalls()
      x, y = gameState.getAgentState(self.index).getPosition()
      vx, vy = Actions.directionToVector(action)  # vektor pomeraja
      newx = int(x + vx)  # nova pozicija x
      newy = int(y + vy)  # nova pozicija y

      # Get set of invaders and defenders
      enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
      invaders = [a for a in enemies if not a.isPacman and a.getPosition() != None]  # napadaju nas duhovi, mi pacman
      defenders = [a for a in enemies if a.isPacman and a.getPosition() != None]  # mi duhovi njih napadamo pacmane

      myState = successor.getAgentState(self.index)
      myPos = myState.getPosition()

      #kad smo presli preko i bezimo od neprijatelja, MORAMO DA GA VRATIMOOOOOOO KUCI
      if (myState.isPacman):
        if myState.numCarrying >= 5:
          newFood = self.getFoodYouAreDefending(gameState).asList()
          minDist =  [self.getMazeDistance(myPos, a) for a in newFood]
        #  features['distanceToFood'] = min(minDist)
          features['backToSafeZone'] = min(minDist)
          features['normalGhosts'] = 1 # NE JEDI
         # print(features["distanceToFood"])
          #return features
        else:
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
          features['ghostInvaders'] = min(dists)
          for ghost in invaders:
            ghostpos = ghost.getPosition()
            neighbors = Actions.getLegalNeighbors(ghostpos, walls)
            if (newx, newy) == ghostpos and ghost.scaredTimer == 0:
              features["normalGhosts"] = 1 #pojesce nas normalan duh, ne idemo na hranu to smo vec resili
          #    features["run"] += 100
            elif (newx, newy) == ghostpos and ghost.scaredTimer > 0:
              features["normalGhosts"] = 0
              features["eatFood"] = 1
            elif ((newx, newy) in neighbors) and (ghost.scaredTimer == 0):
              features["normalGhosts"] = 0.5
           #   features["run"] += 100
            elif ((newx, newy) in neighbors) and ghost.scaredTimer > 0:
              features["normalGhosts"] = 0 #ignorisemo situaciju, idemo ka hrani!!!!

      #kad smo duh, jurimo neprijatelja ako nismo uplaseni
      if not myState.isPacman:
        for pacman in defenders:
          pacmanpos = pacman.getPosition()
          neighbors = Actions.getLegalNeighbors(pacmanpos, walls)
          #features["normalGhosts"] = 0
          features["normalGhosts"] = 1  # ignorisemo situaciju
          #features["eatFood"] = 50  # ako nam neprijatelj u komsiluku, idmeo da ga POJEDEMO, potencijalno dobar potez
          dists = [self.getMazeDistance(myPos, a.getPosition()) for a in defenders]
          features['invaderDistance'] = min(dists)

          if (newx, newy) == pacmanpos and myState.scaredTimer == 0: #JEDEMO
            features["eatFood"] = 100
            features["normalGhosts"] = 1
          elif (newx, newy) == pacmanpos and myState.scaredTimer > 0: #idemo prema hrani da bismo pobegli od pacmana
            features["normalGhosts"] = 1 # bila 0
            features["scaredGhost"] = 1
          elif ((newx, newy) in neighbors) and (myState.scaredTimer == 0):
            #features["normalGhosts"] = 0
            features["normalGhosts"] = 1 # ignorisemo situaciju
            features["eatFood"] = 100 #ako nam neprijatelj u komsiluku, idmeo da ga POJEDEMO, potencijalno dobar potez
          elif ((newx, newy) in neighbors) and myState.scaredTimer > 0:
            features["normalGhosts"] = 1 #ignorisemo situaciju, bila 0
            features["scaredGhost"] = 1


      #na pocetku smo duhovi, nismo uplaseni, nemamo hranu blizu, nemamo neprijatelje, cilj je da predjemo preko
      #trazimo najblizu hranu!!!
      if not features["normalGhosts"]:
        if food1[newx][newy]:
          features["eatFood"] = 1.0
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
            features["distanceToFood"] = float(min(mazedist)) / walldimensions

      #treba da ga vratimo kuci
      return features

    def getWeights(self, gameState, action):
      return {'normalGhosts':-20, 'distanceToFood': -1, 'eatFood': 1, 'invaderDistance': -10, 'run': -10,
              'ghostInvaders': 0.00195, 'scaredGhost': -20, 'backToSafeZone': -1, 'reverse': 10}


class DefensiveAgent(CaptureAgent):

  def registerInitialState(self, gameState):
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)

  def chooseAction(self, gameState):
    """
    Picks among the actions with the highest Q(s,a).
    """
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    return random.choice(bestActions)

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

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)
    return features * weights

  def getFeatures(self, gameState, action):

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    """
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    food1 = self.getFood(gameState)
    capsules = gameState.getCapsules()
    foodList = food1.asList()
    walls = gameState.getWalls()
    x, y = gameState.getAgentState(self.index).getPosition()
    vx, vy = Actions.directionToVector(action)  # vektor pomeraja
    newx = int(x + vx)  # nova pozicija x
    newy = int(y + vy)  # nova pozicija y

    # Get set of invaders and defenders
    enemies = [gameState.getAgentState(a) for a in self.getOpponents(gameState)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]  # invidersi su pacmani
    defenders = [a for a in enemies if not a.isPacman and a.getPosition() != None]  # du duhovi

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()
    # ako nema invidersa mi prelazimo na drugu stranu da jedemo
    if(not myState.isPacman):
      if(len(invaders) == 0 and len(foodList) > 0):
        #idi ka hrani
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
          features["distanceToFood"] = float(min(mazedist)) / walldimensions
"""
    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2, 'distanceToFood': -1}





