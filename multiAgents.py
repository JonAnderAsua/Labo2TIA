# multiAgents.py
# --------------
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


from typing import GenericMeta
from pacman import GameState
from util import manhattanDistance
from game import Actions, Directions
import random, util
import sys

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        # Listas de comida para comparar
        newFoodList = newFood.asList()
        oldFoodList = currentGameState.getFood().asList()

        # score counter
        score = 0
        minfood = sys.maxsize

        # Listas de capsulas para comparar
        oldCapsules = currentGameState.getCapsules()
        newCapsules = successorGameState.getCapsules()

        # Lista de distancias de comida y fantasmas
        foodDist = [manhattanDistance(food, newPos) for food in newFoodList]
        ghostDist = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates]

        #Si el Pacman se come una comida subir el score
        if (len(newFoodList) < len(oldFoodList)):
            score += 1000

        # Si el Pacman se come una capsula subir el score
        if (len(newCapsules) < len(oldCapsules)):
            score += 10000

        # Si el pacman gana que devuelva el maximo
        if successorGameState.isWin():
            return sys.maxsize

        # A penalizar si está quieto
        if action == 'Stop':
           score -= 100

        # Si el pacman está muy cerca hay que bajar el score
        for oneGhostDist in ghostDist:
            if oneGhostDist < 3: # Haciendo pruebas
                score -= 1000000

        #Calcular la distancia minima a la comida
        for oneFoodDist in foodDist:
            if oneFoodDist < minfood:
                minfood = oneFoodDist

        score += 10000 - minfood

        # RETURN TOTAL SCORE
        return score
        #return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """


    def maximo(self,gameState):
        action = gameState.getLegalActions(0)#no index perchè guardo un solo pacman
        v = 1 - sys.maxValue
        for a in action:
            successor= gameState.generateSuccessor(0,a)#a azione della lista
            v = max(v,value(successor,1))

    def minimo(self,gameState,index):
        action = gameState.getLegalActions(index)#molti fantasmi
        v = sys.maxValue
        for a in action:
            successor= gameState.generateSuccessor(0,a)#a azione della lista
            v = max(v,value(successor,1))

    def value(self,gameState,index,depth):#aggiungo self?
        action = gameState.getLegalActions(index)
        agents = gameState.getNumAgents()

        if index == 0:
            depth += 1

        if index > agents:
            index = 0  #perchè non posso prendere più personaggi rispetto a quelli presenti nel gioco

        if not action or  depth == self.depth() + 1:
            return self.evaluationFunction(gameState)
        elif index == 0:
            self.maximo(gameState)
        else:
            self.minimo(gameState,index)

    def getAction(self, gameState):

        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        self.maximo(gameState)
        
    

        

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def maxValue(self, state, alpha, beta, agente):
        v = 1 - sys.maxsize #No sé como poner el - infinito así que por ahora se queda así
        for accion in state.getLegalActions(0): #Con el 0 se coge la posición del pacman
            sucesor = state.generateSuccessor(0,accion)
            v = max(v, self.minValue(sucesor,alpha,beta, agente + 1))
            if v >= beta:
                return v
            alpha = max(alpha,v)
        return v

    def minValue(self, state, alpha, beta, agente):
        v = sys.maxsize #No sé como poner el - infinito así que por ahora se queda así
        for accion in state.getLegalActions(0): #Con el 0 se coge la posición del pacman
            sucesor = state.generateSuccessor(accion,1)
            v = min(v, self.maxValue(sucesor,alpha,beta, agente + 1))
            if v <= alpha:
                return v
            beta = min(beta,v)
        return v

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        alpha = 1 - sys.maxsize
        beta = sys.maxsize
        return self.maxValue(gameState,alpha,beta,0)


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
