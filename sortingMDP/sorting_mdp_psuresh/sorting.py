import numpy as np
import utils
import utils3
import models
from scipy import full, sparse
import generator
import solver
import copy
import math as m
import time

''' Picked/AtHome - means Sawyer is in hover plane at home position
    Placed - means placed back on conveyor after inspecting and finding status as good
    onionLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome', 4: 'Placed'}
    eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    predictions = {0: 'Bad', 1: 'Good', 2: 'Unknown'}
    listIDstatus = {0: 'Empty', 1: 'Not Empty', 2: 'Unavailable'} 
    actList = {0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 2: 'PlaceInBin', 3: 'Pick', 
        4: 'ClaimNewOnion', 5: 'InspectWithoutPicking', 6: 'ClaimNextInList'} '''

def init(nOnionLoc, nEEFLoc, nPredict, nlistIDStatus, discount, useSparse, noise=0.05):

    # nS = nOnionLoc*nEEFLoc*nPredict*nlistIDStatus + 1   # Added one impossible state
    nS = nOnionLoc*nEEFLoc*nPredict*nlistIDStatus
    nA = 7
    # 4 combinations of (predicting good/bad and putting it on conveyor/bin) + 1 to stay still
    # + 1 to claim new onion + 1 to create a new list + 1 to pick a good sorting cycle

    nF = 7
    T = np.zeros((nS, nS, nA))    # state transition probability
    F = np.zeros((nS, nF))         # state feature
    start = np.zeros((nS, 1))
    # T[-1,-1,:] = 1
    # for s in range(nS - 1):
    for s in range(nS):
        onionLoc, eefLoc, pred, listidstatus, start = utils3.sid2vals(s, nOnionLoc, nEEFLoc, nPredict, nlistIDStatus, start)
        actidx = utils3.getValidActions(onionLoc, eefLoc, pred, listidstatus)
        f = np.zeros((nF, 1))
        for a in range(nA):
            nextStates = utils3.findNxtStates(onionLoc, eefLoc, pred, listidstatus, a)
            for nxtS in nextStates:
                ns = utils3.vals2sid(nxtS[0], nxtS[1], nxtS[2], nxtS[3], nOnionLoc, nEEFLoc, nPredict, nlistIDStatus)
                # if ns == nS - 1:
                #     break
                # if s == 78 and a == 2:
                #     print("Hey")
                if a not in actidx:
                    if not (utils3.isValidNxtState(a, nxtS[0], nxtS[1], nxtS[2], nxtS[3])):
                        # Valid actions in invalid current and next states become a sink
                        T[ns, s, a] = 1
                    else:
                        # T[-1, s, a] = 1   # If next state is valid
                        T[91, s, a] = 1   # If next state is valid
                else:
                    if not (utils3.isValidState(onionLoc, eefLoc, pred, listidstatus, s, ns)):  # Invalid actions
                        if not (utils3.isValidNxtState(a, nxtS[0], nxtS[1], nxtS[2], nxtS[3])):
                            # Invalid actions in any current and next states become a sink
                            T[ns, s, a] = 1
                        else:
                            # T[-1, s, a] = 1   # If next state is valid
                            T[91, s, a] = 1   # If next state is valid
                    else:
                        # Valid action in a valid state leading to a valid next state also has a
                        # small failure rate given by noise.
                        if T[s, s, a] == 0:
                            T[s, s, a] = (noise)
                        # Succeding in intended action with high prob
                        T[ns, s, a] += (1 - noise)/len(nextStates)
                
                # Calculate features
                if pred == 1 and nxtS[0] == 4:
                    f[0] = 1
                    
                if pred == 0 and nxtS[0] == 4:
                    f[1] = 1
                    
                if pred == 1 and nxtS[0] == 2:
                    f[2] = 1
                    
                if pred == 0 and nxtS[0] == 2:
                    f[3] = 1
                    
                # Stay still
                if (s == ns):
                    f[4] = 1

                # Claim new onions
                if pred == 2 and onionLoc == 2 or onionLoc == 4 and nxtS[1] == 3:
                    f[5] = 1

                # Fill the list
                if listidstatus == 0 and nxtS[3] == 1:
                    f[6] = 1

                # Pick if unknown
                if onionLoc == 0 and pred == 2 and nxtS[2] == 2 and nxtS[0] == 3:
                    f[7] = 1

                # # Pick after rolling
                # if pred == 0 and listidstatus == 1 and nxtS[0] == 3:
                #     f[8] = 1
 
        F[s, :] = np.transpose(f)

    # Check transition probability
    for a in range(nA):
        for s in range(nS):
            err = abs(sum(T[:, s, a]) - 1)
            # if s == 175:
            #     print(f"T(:,{s},{a}) = {T[:, s, a]}")
            if err > 1e-6 or np.any(T) > 1 or np.any(T) < 0:
                print(f"T(:,{s},{a}) = {T[:, s, a]}")
                print('ERROR: \n', s, a, np.sum(T[:, s, a]))

    start = start / np.sum(start)
    mdp = models.mdp()
    mdp.name = 'sorting'
    mdp.nStates = nS
    mdp.nActions = nA
    mdp.nFeatures = nF
    mdp.discount = discount
    mdp.useSparse = useSparse
    mdp.start = start
    mdp.F = np.tile(F, (nA, 1))
    mdp.transition = T
    mdp.weight = np.zeros((nF, 1))
    mdp.reward = np.reshape(np.dot(mdp.F, mdp.weight), (nS, nA))
    mdp.useSparse = useSparse
    mdp.sampled = False

    if mdp.useSparse:
        mdp.transitionS = {}
        mdp.rewardS = {}
        mdp.F = sparse.csr_matrix(mdp.F)
        mdp.weight = sparse.csr_matrix(mdp.weight)
        mdp.start = sparse.csr_matrix(mdp.start)

        for a in range(mdp.nActions):
            mdp.transitionS[a] = sparse.csr_matrix(mdp.transition[:, :, a])
            mdp.rewardS[a] = sparse.csr_matrix(mdp.reward[:, a])

    return mdp