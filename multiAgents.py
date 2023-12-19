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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent:
    def getAction(self, gameState):
        """
        Get the legal actions available for Pacman.
        Then, choose the best action through evaluating 
        using the evaluation function.
        """
        legalMoves = gameState.getLegalActions() #list of legal actions that the pacman can take 
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves] #evaluate each legal action using evaluation function 
        bestScore = max(scores) #get the best score from the evaluated actions 
        bestIndices = [index for index, score in enumerate(scores) if score == bestScore] #get indices of the action with best score 
        chosenIndex = random.choice(bestIndices)  #select an action randomly which has best score 
        
        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design an evaluation function to estimate the utility of a state-action pair.
        """
        # Get successor game state
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        
        newPos = successorGameState.getPacmanPosition() #New Position of Pacman 
        newFood = successorGameState.getFood() #location of food in the successor game state 
        newGhostStates = successorGameState.getGhostStates() #Ghost state in success game state 
        
        ghostDistances = [manhattanDistance(newPos, ghost.getPosition()) for ghost in newGhostStates] #distance from pacman to each of the ghost in sucessor game state 
        minGhostDistance = min(ghostDistances) if ghostDistances else 1 #find smallest distance, if there are no distance set to 1 
        
        foodDistances = [manhattanDistance(newPos, food) for food in newFood.asList()] #distance between pacman and each food location 
        minFoodDistance = min(foodDistances) if foodDistances else 1 #find smallest distance, if there are no distance set to 1 
        
        ghostScore = 1.5/minGhostDistance if minGhostDistance > 0 else 0 #if nearest ghost is very near then score is high
        foodScore = 1.0/minFoodDistance if minFoodDistance > 0 else 0 #if nearest ghost is very near then score is high
        
        return successorGameState.getScore() + foodScore - ghostScore
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

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        """
        # Start the Minimax algorithm from Pacman's perspective (maximizing agent).
        # Maximize for agent 0 which is pacman 
        value, action = self.maxValue(gameState, 0, 0)
        return action
    
    def maxValue(self, gameState, depth, agentIndex):
        """
        Compute the max value for the given state, depth, and agent.
        """
        #if the game is over or the depth limit is reached, evaluate the state.
        if gameState.isWin() or gameState.isLose() or depth == self.depth:
            return self.evaluationFunction(gameState), ""
        # Initialize value to negative infinity and action to an empty string.
        value = float("-inf")
        action = ""
         # Iterate over all possible actions for Pacman
        for a in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, a)
            # For each action, determine the utility value from the ghosts' perspective.
            successorValue, _ = self.minValue(successorState, depth, 1)
            # Update the value and action if a better one is found.
            if successorValue > value:
                value, action = successorValue, a
        return value, action
    
    def minValue(self, gameState, depth, agentIndex):
        """
        Compute the min value for the given state, depth, and agent.
        """
        #if game is over or depth has been reached ten evaluate the state 
        if gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), ""
        
        # Initialize value to positive infinity and action to an empty string.
        value = float("inf")
        action = ""
        
        # Iterate over all possible actions 
        for a in gameState.getLegalActions(agentIndex):
            successorState = gameState.generateSuccessor(agentIndex, a)
            # If the current ghost is the last one, increase the depth and switch to Pacman's perspective.
            if agentIndex == gameState.getNumAgents() - 1:  # Last ghost
                successorValue, _ = self.maxValue(successorState, depth + 1, 0)
            else:  # If there are more ghosts left in the current depth level
                successorValue, _ = self.minValue(successorState, depth, agentIndex + 1)
                # Update the value and action if a lower utility value is found (since it's a minimizing node).
            if successorValue < value:
                value, action = successorValue, a
        return value, action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        def maxValue(state, depth, alpha, beta):
            # Terminal test: if the game is over or the depth limit is reached, evaluate the state.
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), ""
            v = float("-inf")
            best_action = ""
            for action in state.getLegalActions(0): # 0 represents Pacman (maximizing agent)
                successorValue, _ = minValue(state.generateSuccessor(0, action), depth, 1, alpha, beta)
                if successorValue > v:
                    v, best_action = successorValue, action
                if v > beta: # Prune the remaining actions
                    return v, action
                alpha = max(alpha, v) # Update the alpha value
            return v, best_action
        
        def minValue(state, depth, agentIndex, alpha, beta):
            # Terminal test: if the game is over, evaluate the state.
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state), ""
            v = float("inf")
            best_action = ""
            for action in state.getLegalActions(agentIndex):
                if agentIndex == state.getNumAgents() - 1: #Last Ghost 
                    successorValue, _ = maxValue(state.generateSuccessor(agentIndex, action), depth + 1, alpha, beta)
                else: #more ghost at current depth level 
                    successorValue, _ = minValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1, alpha, beta)
                if successorValue < v:
                    v, best_action = successorValue, action
                if v < alpha: #prune remaining actions 
                    return v, action
                beta = min(beta, v) #update beta 
            return v, best_action
        
            # Start the alpha-beta search from Pacman's perspective
        _, action = maxValue(gameState, 0, float("-inf"), float("inf"))
        return action

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        """
        def maxValue(state, depth):
             # Terminal test: if the game is over or the depth limit is reached, evaluate the state.
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state), ""
            v = float("-inf")
            best_action = ""
            for action in state.getLegalActions(0): # 0 represents Pacman (maximizing agent)
                successorValue, _ = expValue(state.generateSuccessor(0, action), depth, 1)
                if successorValue > v:
                    v, best_action = successorValue, action
            return v, best_action
        
        def expValue(state, depth, agentIndex):
             # Terminal test: if the game is over, evaluate the state.
            if state.isWin() or state.isLose():
                return self.evaluationFunction(state), ""
            v = 0
            best_action = ""
            prob = 1.0 / len(state.getLegalActions(agentIndex))  # Probability for each action, assuming uniform distribution
            for action in state.getLegalActions(agentIndex):
                if agentIndex == state.getNumAgents() - 1:
                    successorValue, _ = maxValue(state.generateSuccessor(agentIndex, action), depth + 1)
                else: # More ghosts remain at the current depth level
                    successorValue, _ = expValue(state.generateSuccessor(agentIndex, action), depth, agentIndex + 1)
                v += prob * successorValue  # Calculate the expected value by weighting the successor values by their probabilities
            return v, best_action
         # Start the Expectimax search from Pacman's perspective
        _, action = maxValue(gameState, 0)
        return action
