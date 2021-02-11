#!/usr/bin/env python
import sys
import Queue
import subprocess
import multiprocessing
import random
import cPickle as pickle
import os
import operator
import time
import numpy as np
import util.classes
# from patrol.model import boyd2MapParams, OGMap, PatrolModel 
from sortingMDP.model import sortingModel,InspectAfterPicking,\
PlaceOnConveyor,PlaceInBin,Pick,ClaimNewOnion,InspectWithoutPicking,\
ClaimNextInList,sortingState 
from sortingMDP.model import sortingModel2,\
PlaceInBinClaimNextInList,sortingModelbyPSuresh,\
sortingModelbyPSuresh2,sortingModelbyPSuresh3,\
sortingModelbyPSuresh4,sortingModelbyPSuresh2WOPlaced,\
sortingModelbyPSuresh3multipleInit

from sortingMDP.reward import sortingReward2,\
sortingReward3,sortingReward4,sortingReward5,\
sortingReward6,sortingReward7,sortingReward7WPlaced

from mdp.solvers import *
import mdp.agent
from mdp.simulation import *
import re

import rospy
from std_msgs.msg import Int32MultiArray
from sorting_patrol_MDP_irl.srv import requestPolicy

home = os.environ['HOME']
def get_home():
    global home
    return home

use_frequentist_baseline = True
inputDcode = str(use_frequentist_baseline)+"\n"
useHierDistr = True
inputDcode += str(useHierDistr)+"\n"
lbfgs_use_ones = False
inputDcode += str(lbfgs_use_ones)+"\n"

NumObFeatures = 12
if useHierDistr: 
    trueDistr_obsfeatures = [0]*2*NumObFeatures
else:
    trueDistr_obsfeatures = [0]*NumObFeatures

totalmass_untilNw = 0.0
for i in range(NumObFeatures):
    if useHierDistr: 
        currp = random.uniform(0.0,.99) 
    else:
        currp = uniform(0.0,1-totalmass_untilNw)
        totalmass_untilNw += currp
    
    trueDistr_obsfeatures[i] = currp 
    if useHierDistr: 
        trueDistr_obsfeatures[i+NumObFeatures] = 1-currp 

if not useHierDistr: 
    trueDistr_obsfeatures[NumObFeatures-1] += 1-totalmass_untilNw 

inputDcode += str(trueDistr_obsfeatures)+"\n"

#for I2RL
numSessionsSoFar = 0
num_Trajsofar = 0
reward_dim = 11
learned_mu_E=[0.0]*reward_dim
learned_weights=[0.0]*reward_dim
lineFoundWeights=""
lineFeatureExpec=""

if (useHierDistr == 1): 
    runAvg_learnedDistr_obsfeatures = [0]*2*NumObFeatures
else:
    runAvg_learnedDistr_obsfeatures = [0]*NumObFeatures

f_inputDcode = open(get_home() + "/Downloads/inputDcode.txt", "w")
f_inputDcode.write("")
f_inputDcode.close()
# f_outputDcode = open(get_home() + "/Downloads/outputDcode.txt", "w")
# f_outputDcode.write("")
# f_outputDcode.close()
# f_outputDcode = open(get_home() + "/Downloads/outputDcode.txt", "w")

##############################################################
##############################################################

''' 
    Picked/ AtHome - means Sawyer is in hover plane at home position

    onionLoc = {0: 'OnConveyor', 1: 'InFront',
    2: 'InBin', 3: 'Picked/AtHome'}
    
    eefLoc = {0: 'OnConveyor', 1: 'InFront', 2: 'InBin', 3: 'Picked/AtHome'}
    
    predictions = {0: 'Bad', 1: 'Good', 2: 'Unknown'}
    
    listIDstatus = {0: 'Empty', 1: 'Not Empty', 2: 'Unavailable'}
    
    actList = {0: 'InspectAfterPicking', 1: 'PlaceOnConveyor', 2: 'PlaceInBin', 3: 'Pick',
    4: 'ClaimNewOnion', 5: 'InspectWithoutPicking', 6: 'ClaimNextInList'} 

''' 

nOnionLoc=5
nEEFLoc=4
nPredict=3
nlistIDStatus=3
array_pol=[4]*nOnionLoc*nEEFLoc*nPredict*nlistIDStatus

act2aid = { 'InspectAfterPicking':0,
            'PlaceOnConveyor':1,
            'PlaceInBin':2,
            'Pick':3,
            'ClaimNewOnion':4,
            'InspectWithoutPicking':5,
            'ClaimNextInList':6       
        } 

def state2sid(ol, eefl, pred, listst, nOnionLoc=5, nEEFLoc=4, nPredict=3, nlistIDStatus=3):
    return(ol + nOnionLoc * (eefl + nEEFLoc * (pred + nPredict * listst)))

def parse_sorting_policy_encode_array(buf):
    global array_pol
    print("len(array_pol): ",len(array_pol))
    stateactions = buf.split("\n")
    for stateaction in stateactions:
        temp = stateaction.split(" = ")
        if len(temp) < 2: continue
        state = temp[0]
        action = temp[1]
                                                
        state = state[1 : len(state) - 1]
        pieces = state.split(",")	
        
        ol, pred, eefl, listst = int(pieces[0]), int(pieces[1]), int(pieces[2]), int(pieces[3])
        # print("ol {}, pred {}, eefl {}, listst {}".format(ol, eefl, pred, listst))
        ss = state2sid(ol, eefl, pred, listst)

        if action in act2aid:
            act = act2aid[action]
        else:
            print("parse_sorting_policy: input action {} not in dictionary ".format(action))
            exit(0)
        
        # print("parsed ss {} a {}".format(ss,act)) 
        array_pol[ss] = act
    return

##############################################################
###############################################################

def cb_runRobustIrlGetPolicy(msg):
    # service call back running a session and returning learned policy
    global inputDcode, numSessionsSoFar, num_Trajsofar, reward_dim, learned_mu_E, \
    lineFoundWeights, lineFeatureExpec, runAvg_learnedDistr_obsfeatures

    variablepart_inputDcode = ""
    # input to D code: weights, feature expectations 
    if numSessionsSoFar == 0: 

        for j in range(reward_dim): 
            learned_weights[j] = random.uniform(0.0,.99)
        
        lineFoundWeights = str(learned_weights) #+"\n"
        # create initial feature expectations
        for j in range(reward_dim):
            learned_mu_E[j]=0.0
        
        lineFeatureExpec = str(learned_mu_E) #+"\n"

    if not not lineFoundWeights and lineFoundWeights[-1] != '\n':
        lineFoundWeights = lineFoundWeights + "\n" 

    if not not lineFeatureExpec and lineFeatureExpec[-1] != '\n': 
        lineFeatureExpec = lineFeatureExpec + "\n" 

    variablepart_inputDcode += lineFoundWeights+lineFeatureExpec+ str(num_Trajsofar)+"\n"  
    variablepart_inputDcode += str(numSessionsSoFar)+"\n"+str(runAvg_learnedDistr_obsfeatures)+"\n"

    print ("input to Robust IRL solver \n",inputDcode + variablepart_inputDcode)
    f_inputDcode = open(get_home() + "/Downloads/inputDcode.txt", "w")
    f_inputDcode.write(inputDcode + variablepart_inputDcode)
    f_inputDcode.close()
    # exit(0)

    args = [get_home() +"/catkin_ws/devel/bin/"+ "runSessionUnknownObsModRobustIRL", ]
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True)				
    (stdout, stderr) = p.communicate(inputDcode + variablepart_inputDcode) 
    
    # f_outputDcode.write(stdout)
    # f_outputDcode.close()
    # exit(0)

    # f_outputDcode = open(get_home() + "/Downloads/outputDcode.txt", "r")
    # stdout = f_outputDcode.read()
    # f_outputDcode.close()
    # print (stdout)

    contentExists = len(re.findall('BEGPARSING([.\s\S]*)ENDPARSING', stdout))!=0
    if contentExists:

        LBA = re.findall('LBA([.\s\S]*)ENDLBA', stdout)[0]
        print LBA
        # DIFF1 = re.findall('DIFF1([.\s\S]*)ENDDIFF1', stdout)[0]
        # print DIFF1
        # DIFF2 = re.findall('DIFF2([.\s\S]*)ENDDIFF2', stdout)[0]
        # print DIFF2
        lineFoundWeights = re.findall('WEIGHTS([.\s\S]*)ENDWEIGHTS', stdout)[0]
        # print lineFoundWeights
        lineFeatureExpec = re.findall('FE([.\s\S]*)ENDFE', stdout)[0]
        # print lineFeatureExpec
        num_Trajsofar = re.findall('NUMTR([.\s\S]*)ENDNUMTR', stdout)[0]
        # print num_Trajsofar
        numSessionsSoFar = re.findall('NUMSS([.\s\S]*)ENDNUMSS', stdout)[0]
        # print numSessionsSoFar
        runAvg_learnedDistr_obsfeatures = re.findall('RUNAVGPTAU([.\s\S]*)ENDRUNAVGPTAU', stdout)[0]
        # print runAvg_learnedDistr_obsfeatures
        policyString = re.findall('POLICY([.\s\S]*)ENDPOLICY', stdout)[0]
        # print policyString 
        global array_pol
        print("policyString ",policyString)
        parse_sorting_policy_encode_array(policyString)
        print("array_pol ",array_pol)
            
    else:
        print ("Bad Run: Optimization stopped half way, Try again. ")
        print ("contentExists"+str(contentExists))

    return array_pol

if __name__ == "__main__": 

    ''' 
    decide distribution over features and keep it fixed. 
    the number of features are decided within D code, hard code that part

    for each session 
    sample data within D code
    run session 
    parse returned results
    convert policy to encoded array
    send it through pub-sub 

    '''
    rospy.init_node('run_session_update_policy', anonymous=True)
    runRobustIrlSessionService = rospy.Service("/runRobustIrlGetPolicy", requestPolicy, cb_runRobustIrlGetPolicy)
    rospy.spin()

