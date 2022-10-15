'''
Created on Sep 09, 2022

@author: Samira Fallah (saf418@lehigh.edu)
'''

from pulp import LpVariable, lpSum, LpInteger, LpStatusOptimal, LpProblem
from pulp import LpMinimize, LpMaximize,LpStatus, pulp
import random
from pyomo.environ import *
import numpy as np
import time

numVars = 11
numIntVars = 10
numConsFixed = 1
numConsVar = 1
INTVARS = range(numIntVars)
SLACKVARS = range(numIntVars, numVars)
CONSVARRHS = range(numConsVar)
CONSFIXEDRHS = range(numConsFixed)

timeLimit = 86400
debug_print = True

OBJ = [-566, -611, -506, -180, -817, -184, -585, -423, -26, -317, 0]
MAT = {(0, 0):-62, (0, 1):-84, (0, 2):-977, (0, 3):-979, (0, 4):-874, (0, 5):-54, (0, 6):-269, (0, 7):-93, (0, 8):-881, (0, 9):-563, (0, 10):0}
MATFixed = {(0, 0):557, (0, 1):898, (0, 2):148, (0, 3):63, (0, 4):78, (0, 5):964, (0, 6):246, (0, 7):662, (0, 8):386, (0, 9):272, (0, 10):1}
RHS = {0:2137}

listTemp = [] 
Master = AbstractModel()
Master.intIndices = Set(initialize=INTVARS)
Master.slackIndices = Set(initialize=SLACKVARS)
Master.constraintSetVarRHS = Set(initialize=CONSVARRHS)
Master.constraintSetFixedRHS = Set(initialize=CONSFIXEDRHS)
Master.intPartList = Set(initialize=listTemp)
Master.dualVarSetVarRHS = Master.constraintSetVarRHS * Master.intPartList
Master.theta = Var(domain=Reals, bounds = (None, None))
Master.intVars = Var(Master.intIndices, domain=Binary, bounds=(0, 1))
Master.slackVars = Var(Master.slackIndices, domain=NonNegativeIntegers, bounds=(0, None))
Master.dualVarsVarRHS = Var(Master.dualVarSetVarRHS, domain=Reals, bounds=(None, 0))

def objective_rule(model):
    return model.theta
Master.objective = Objective(rule=objective_rule, sense=maximize)

def theta_constraint_rule(model, k):
    return (model.theta <=
            sum(OBJ[j]*(model.int_part_list[k][j] - model.intVars[j]) for j in INTVARS)
            + sum(MAT[(i, j)]*model.dualVarsVarRHS[(i, k)]*(model.intVars[j] - model.int_part_list[k][j]) 
                for j in INTVARS for i in CONSVARRHS))
      
Master.theta_constraint = Constraint(Master.intPartList, rule=theta_constraint_rule)

def UB_constraint_rule(model):
    return (model.theta + sum(OBJ[j]*model.intVars[j] for j in INTVARS) <= U)

Master.UB_constraint = Constraint(rule=UB_constraint_rule)

def fixed_constraint_rule(model, k):
    return (sum(MATFixed[(k, j)]*model.intVars[j] for j in INTVARS) + sum(MATFixed[(k, j)]*model.slackVars[j] for j in SLACKVARS) == RHS[k])

Master.fixed_constraint = Constraint(Master.constraintSetFixedRHS, rule=fixed_constraint_rule)

def changeValue(value):
    if str(value) == 'None':
        return 0.0
    return value

# Generate a feasible solution to start with
def generateInitPoint():
    prob = LpProblem("IP", LpMaximize)

    intVarsInit = LpVariable.dicts("int", INTVARS, lowBound = 0, upBound = 1, cat='Binary')
    slackVarsInit = LpVariable.dicts("slack", SLACKVARS, lowBound = 0, upBound = None, cat='Integer')

    prob += lpSum(intVarsInit[i] * OBJ[i] for i in INTVARS) 

    for i in range(len(RHS.keys())):
        prob += lpSum(intVarsInit[j] * MATFixed[(i, j)] for j in INTVARS) + lpSum(slackVarsInit[j] * MATFixed[(i, j)] for j in SLACKVARS) == RHS[i]

    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    return {k : changeValue(intVarsInit[k].varValue) for k in INTVARS}

start = time.time()  
intVarsInit = generateInitPoint()
U = sum(OBJ[j]*intVarsInit[j] for j in INTVARS)

# Convert the feasible solution to an NDP
def convertWeakToStrongNDP(_intVarsInit, _print=False):
    prob = LpProblem("IPConvertToNDP", LpMinimize)

    intVarsStrong = LpVariable.dicts("int", INTVARS, lowBound = 0, upBound = 1, cat='Binary')
    slackVarsStrong = LpVariable.dicts("slack", SLACKVARS, lowBound = 0, upBound = None, cat='Integer')

    prob += (sum(OBJ[i] * intVarsStrong[i] for i in INTVARS) 
                                    + sum(MAT[(i, j)] * intVarsStrong[j] for j in INTVARS for i in CONSVARRHS))
    
    RHSFirstObj = 0
    for i in INTVARS:
        RHSFirstObj += OBJ[i] * _intVarsInit[i]
      
    prob += sum(OBJ[i] * intVarsStrong[i] for i in INTVARS) <= RHSFirstObj
    
    for j in CONSVARRHS:
        prob += (sum(MAT[(j, i)] * intVarsStrong[i] for i in INTVARS) <= sum(MAT[(j, i)] * _intVarsInit[i] for i in INTVARS))
        
    for j in range(len(RHS.keys())):
        prob += (sum(MATFixed[(j, i)] * intVarsStrong[i] for i in INTVARS) 
                                + sum(MATFixed[(j, i)] * slackVarsStrong[i] for i in SLACKVARS) == RHS[j])
    
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    if _print:
        with open("Results_IP_details.txt", "a") as _filed:
            _filed.write('Solution in iteration 0' + '\n\n') 
            _filed.write("Integer Variables:\n")
            for j in INTVARS:
                _filed.write(str(j) + ' ' + str(round(changeValue(intVarsStrong[j].varValue))) + '\n')
            _filed.write('\n' + "Slack Variables:\n")  
            for j in SLACKVARS:
                _filed.write(str(j) + ' ' + str(round(changeValue(slackVarsStrong[j].varValue))) + '\n')
            _filed.write('-------------------------------------' + '\n')

    temp = {}
    for j in INTVARS:
        temp[j] = changeValue(intVarsStrong[j].varValue)
   
    return temp

intVarsInitStrong = convertWeakToStrongNDP(intVarsInit, _print=True)

Master.int_part_list = [intVarsInitStrong]

EF = []

temp_ndp = ()
temp_ndp = temp_ndp + (sum(OBJ[j]*intVarsInitStrong[j] for j in INTVARS),)

for k in CONSVARRHS: 
    temp_ndp = temp_ndp + (sum(MAT[(k, l)]*intVarsInitStrong[l] for l in INTVARS),)

EF.append(temp_ndp)

opt = SolverFactory("couenne")

idxIntPartList = 0
thetaList = [] 
counterRep = 0
while True:
    listTemp.append(idxIntPartList)
    instance = Master.create_instance()
    #print(instance.pprint())
    results = opt.solve(instance)
    instance.solutions.load_from(results)
    if debug_print:
        with open("Results_IP_details.txt", "a") as _filed:
            _filed.write('Solution in iteration' + ' ' + str(idxIntPartList+1) + '\n\n') 
            _filed.write("Integer Variables:\n")
            for j in instance.intVars:
                _filed.write(str(j) + ' ' + str(round(changeValue(instance.intVars[j].value))) + '\n')
            _filed.write('\n' + "Slack Variables:\n")  
            for j in instance.slackVars:
                _filed.write(str(j) + ' ' + str(round(changeValue(instance.slackVars[j].value))) + '\n')
            _filed.write('\n' + "Dual Variables (varying RHSs):\n")  
            for j in instance.dualVarsVarRHS:
                _filed.write(str(j) + ' ' + str(round(changeValue(instance.dualVarsVarRHS[j].value), 5)) + '\n')
            _filed.write('\n' + "Theta: " + str(round(instance.theta.value, 2))  + '\n')
            _filed.write('-------------------------------------' + '\n')

    end = time.time()
    elapsedTime = end - start
    thetaList.append(round(instance.theta.value, 2))
    
    if instance.theta.value < 0.1 or elapsedTime > timeLimit:
        if instance.theta.value < 0.1:
            print("Finished!")
        else:
            print("Finished due to time limit!")
            
#         zetaValuesList = [[] for k in CONSVARRHS] 
#         valueFuncList = []
    
#         for int_part in Master.int_part_list:
#             for k in CONSVARRHS:
#                 zetaNew = sum(MAT[(k, l)]*int_part[l] for l in INTVARS) 
#                 zetaValuesList[k].append(zetaNew)
#             VFValue = sum(OBJ[j]*int_part[j] for j in INTVARS) 
#             valueFuncList.append(VFValue)

#         EF = [[] for i in range(len(valueFuncList))]
#         for i in range(len(valueFuncList)):
#             EF[i].append(valueFuncList[i])        
#         for j in range(len(valueFuncList)):
#             for i in range(numConsVar):
#                 EF[j].append(zetaValuesList[i][j])

        if instance.theta.value < 0.1:
            with open("Results_IP.txt", "w") as _file:
                _file.write('Efficient Frontier: '+ str(EF) + '\n') 
                _file.write('Elapsed Time: ' + str(round(elapsedTime, 2)) + ' sec' + '\n')
                _file.write('Theta list: ' + str(thetaList))
        else:
            with open("Results_IP.txt", "w") as _file:
                _file.write('Approximate Efficient Frontier: '+ str(EF) + '\n')
                _file.write('Elapsed Time: ' + str(round(elapsedTime, 2)) + ' sec' + '\n')
                _file.write('Theta list: ' + str(thetaList))

        break
    
    int_part = convertWeakToStrongNDP({**dict((i, round(changeValue(instance.intVars[i].value))) for i in INTVARS)}, _print=False)
    
    temp_ndp = () 
    temp_ndp = temp_ndp + (sum(OBJ[j]*int_part[j] for j in INTVARS),)
    
    for k in CONSVARRHS:
        temp_ndp = temp_ndp + (sum(MAT[(k, l)]*int_part[l] for l in INTVARS),) 
        
    if temp_ndp in EF:
        counterRep += 1
        if counterRep >= 2:
            with open("Results_IP.txt", "w") as _file:
                _file.write('Algorithm stopped. Solution is already exist! The theta value ' + str(round(instance.theta.value, 2)) + '\n')
                _file.write('Efficient Frontier: ' + str(EF) + '\n')
                _file.write('Elapsed Time: ' + str(round(elapsedTime, 2)) + ' sec' + '\n')
                _file.write('Theta list: ' + str(thetaList) + '\n')
                _file.write('Number of NDPs: ' + str(len(EF)))
                break
            
    EF.append(temp_ndp)    
    Master.int_part_list.append(int_part)
    
    idxIntPartList += 1