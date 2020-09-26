import vrep, math, sys, time, numpy as np, cv2

vrep.simxFinish(-1)
time.sleep(1)
j1_limit =  np.array([33.0, 93.0]);
loc=[0.45,-0.10,0.53620]
sleepFactor = 1.5 
dim1, dim2 = 32, 64
clientID = 0
clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
if clientID!=-1:
    print 'Connected to remote API server with clientID: ', clientID
    vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
else:
    print 'Failed connecting to remote API server'
    sys.exit(1)        
        
[res0,torso]=vrep.simxGetObjectHandle(clientID, 'torso_11_respondable', vrep.simx_opmode_blocking)

[res1,shoulderZ]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_z', vrep.simx_opmode_blocking)
[res2,shoulderY]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_y', vrep.simx_opmode_blocking)
[res3,armX]=vrep.simxGetObjectHandle(clientID, 'r_arm_x', vrep.simx_opmode_blocking)
[res4,elbowY]=vrep.simxGetObjectHandle(clientID, 'r_elbow_y', vrep.simx_opmode_blocking)
[res5,wristZ]=vrep.simxGetObjectHandle(clientID, 'r_wrist_z', vrep.simx_opmode_blocking)
[res6,wristX]=vrep.simxGetObjectHandle(clientID, 'r_wrist_x', vrep.simx_opmode_blocking)

[res8,handleSensor1]=vrep.simxGetObjectHandle(clientID, 'vision_sensor',vrep.simx_opmode_blocking)

[res9,TableWrist]=vrep.simxGetCollisionHandle(clientID, 'TableWrist',vrep.simx_opmode_blocking)


class Env:    
    def __init__(self):
        self.torso = torso
    
    def connectRobot(self):
        clientID=vrep.simxStart('127.0.0.1',19997,True,True,5000,5)
        if clientID!=-1:
            print 'Connected to remote API server with clientID: ', clientID
            vrep.simxStartSimulation(clientID, vrep.simx_opmode_oneshot_wait)
        else:
            print 'Failed connecting to remote API server'
            sys.exit(1)        
        return clientID

    
    def disconnectRobot(self):
        vrep.simxStopSimulation(clientID, vrep.simx_opmode_oneshot)
        vrep.simxFinish(clientID)
        print 'Program ended'      

 
    #return joint positions in degree
    def getJointConfig(self):        
        pos1 = vrep.simxGetJointPosition(clientID, shoulderZ, vrep.simx_opmode_blocking)
        pos2 = vrep.simxGetJointPosition(clientID, shoulderY, vrep.simx_opmode_blocking)
        pos3 = vrep.simxGetJointPosition(clientID, armX, vrep.simx_opmode_blocking)
        pos4 = vrep.simxGetJointPosition(clientID, elbowY, vrep.simx_opmode_blocking)
        pos5 = vrep.simxGetJointPosition(clientID, wristZ, vrep.simx_opmode_blocking)
        pos6 = vrep.simxGetJointPosition(clientID, wristX, vrep.simx_opmode_blocking)
        
        positions = []
        positions.append(round(pos1[1]/math.pi * 180,1))
        positions.append(round(pos2[1]/math.pi * 180,1))
        positions.append(round(pos3[1]/math.pi * 180,1))
        positions.append(round(pos4[1]/math.pi * 180,1))
        positions.append(round(pos5[1]/math.pi * 180,1))
        positions.append(round(pos6[1]/math.pi * 180,1))

        return positions

    def setJointConfiguration(self, armX_, shoulderY_, elbowY_, shoulderZ_, wristZ_, wristX_):
        [res1,shoulderZ]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_z', vrep.simx_opmode_blocking)
        [res2,shoulderY]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_y', vrep.simx_opmode_blocking)
        [res3,armX]=vrep.simxGetObjectHandle(clientID, 'r_arm_x', vrep.simx_opmode_blocking)
        [res4,elbowY]=vrep.simxGetObjectHandle(clientID, 'r_elbow_y', vrep.simx_opmode_blocking)
        [res5,wristZ]=vrep.simxGetObjectHandle(clientID, 'r_wrist_z', vrep.simx_opmode_blocking)
        [res6,wristX]=vrep.simxGetObjectHandle(clientID, 'r_wrist_x', vrep.simx_opmode_blocking)
        
        vrep.simxSetJointTargetPosition(clientID, armX, math.radians(armX_), vrep.simx_opmode_blocking)
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, shoulderY, math.radians(shoulderY_), vrep.simx_opmode_blocking)
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, elbowY, math.radians(elbowY_), vrep.simx_opmode_blocking)
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, shoulderZ, math.radians(shoulderZ_), vrep.simx_opmode_blocking)
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, wristZ, math.radians(wristZ_), vrep.simx_opmode_blocking)
        time.sleep(sleepFactor)
        vrep.simxSetJointTargetPosition(clientID, wristX, math.radians(wristX_), vrep.simx_opmode_blocking)
        time.sleep(sleepFactor)
 
    def getTargetPosition(self):
        [res7,target]=vrep.simxGetObjectHandle(clientID, 'Target', vrep.simx_opmode_blocking)
        [res, pos] = vrep.simxGetObjectPosition(clientID, target, -1, vrep.simx_opmode_blocking)

        return pos 
    
    def resetGoal(self,loc):
        [res7,target]=vrep.simxGetObjectHandle(clientID, 'Target', vrep.simx_opmode_blocking)
        vrep.simxSetModelProperty(clientID, target, 32+64, vrep.simx_opmode_blocking) 
        vrep.simxSetObjectOrientation(clientID,target,-1,[math.pi, 0., 0.],vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(clientID,target,-1, loc, vrep.simx_opmode_blocking)
        vrep.simxSetModelProperty(clientID, target, 0, vrep.simx_opmode_blocking)
        
    def reset(self,loc=[0.45,-0.10,0.53620]):
        setShoulderZ(33)
        time.sleep(sleepFactor+2.5) 
        self.resetGoal(loc)
        time.sleep(sleepFactor/2.)
    
    def dist2goal(self):
        [res7,Dummy]=vrep.simxGetObjectHandle(clientID, 'Dummy', vrep.simx_opmode_blocking)
        
        tPos = self.getTargetPosition()
        [res, gripperPos] = vrep.simxGetObjectPosition(clientID, Dummy, -1, vrep.simx_opmode_blocking)
        dist = np.linalg.norm(np.array(gripperPos)-np.array(tPos))
        
        return dist 
     
    def targetFell(self):
        while True:
            [res7,target]=vrep.simxGetObjectHandle(clientID, 'Target', vrep.simx_opmode_blocking)
            if res7 ==0: break
        while True:
            [res,ori] = vrep.simxGetObjectOrientation(clientID, target,-1,vrep.simx_opmode_blocking) # in radian
            if res ==0 and not -3.2<= ori[0]<= -2.9: break
        dist = np.linalg.norm(np.array([ori[0],ori[1],0.])-np.array([math.pi,0.,0.]))

        return dist > 0.2 
    
    def _getImage(self): 
        while True:
            err, res, img =  vrep.simxGetVisionSensorImage(clientID, handleSensor1, 0, vrep.simx_opmode_blocking) # 0 for RGB
            if err == 0: break
        img = np.array(img,dtype = np.uint8)
        img = img.reshape((res[1],res[0],3))
        img = np.flipud(img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img

    def writeImage(self,img):
        filename='nico_' + str(time.clock())
        cv2.imwrite("img2/"+filename +".png",img)
        
    def test_for_collsion(self):
        [res, robot_table] = vrep.simxGetCollisionHandle(clientID, 'robot_table',vrep.simx_opmode_blocking)
        
        return vrep.simxReadCollision(clientID, robot_table, vrep.simx_opmode_blocking)[1]
        
    def closeHand(self):
        [res,indexFing]=vrep.simxGetObjectHandle(clientID, 'r_indexfingers_x', vrep.simx_opmode_blocking)
        [res,thumb]=vrep.simxGetObjectHandle(clientID, 'r_thumb_x', vrep.simx_opmode_blocking)
        
        vrep.simxSetJointTargetPosition(clientID, indexFing, math.radians(-50), vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetPosition(clientID, thumb, math.radians(-50), vrep.simx_opmode_blocking)
        
    def openHand(self):
        [res,indexFing]=vrep.simxGetObjectHandle(clientID, 'r_indexfingers_x', vrep.simx_opmode_blocking)
        [res,thumb]=vrep.simxGetObjectHandle(clientID, 'r_thumb_x', vrep.simx_opmode_blocking)
        
        vrep.simxSetJointTargetPosition(clientID, indexFing, math.radians(0), vrep.simx_opmode_blocking)
        vrep.simxSetJointTargetPosition(clientID, thumb, math.radians(0), vrep.simx_opmode_blocking)
        
    def _reward (self):
        self.targetFell()
        if self.targetFell():
            reward = -1.
        else:      
            dist2goal = self.dist2goal()
            if dist2goal <=0.03:
                loc_ = self.getTargetPosition()
                time.sleep(sleepFactor/2)
                # test for successful grasp, then reward with +1 for success. Otherwise, move config to the previously registered joint value 
                self.closeHand()
                time.sleep(sleepFactor/2)
                setShoulderZ(getShoulderZ()-20)
                if self.dist2goal()<0.03:
                    print "success"
                    reward = 1.
                    self.openHand()
                else:
                    self.openHand()
                    time.sleep(0.8)
                    setShoulderZ(getShoulderZ()+20)
                    self.resetGoal(loc_)
                    reward = -(dist2goal**2) 
            else:
                reward = -(dist2goal**2)
        
        return reward
    
    def step(self, action):
        prev_config = getShoulderZ() 
        done = False
        
        joint = action[0] + prev_config

        if joint < j1_limit[0]: joint = j1_limit[0]; 
        elif joint > j1_limit[1]: joint = j1_limit[1];

        obs = self.retrieve(joint)
        reward = self._reward()
        
        if reward == 1. or reward == -1:
            done = True;
        
        return  obs, reward, done
    
    def retrieve (self, joint):
        setShoulderZ(joint)
        img = self._getImage().astype(float); self.writeImage(img);
        img/=256.0 
        img = img.reshape((dim2,dim1,3))

        return img

def setArm(angle): 
    [res3,armX]=vrep.simxGetObjectHandle(clientID, 'r_arm_x', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, armX, math.radians(angle), vrep.simx_opmode_blocking)

def setShoulderZ(angle): 
    [res3,shoulderZ]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_z', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, shoulderZ, math.radians(angle), vrep.simx_opmode_blocking)
    
def setShoulderY(angle): 
    [res3,shoulderY]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_y', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, shoulderY, math.radians(angle), vrep.simx_opmode_blocking)

def setElbow(angle):
    [res3,elbowY]=vrep.simxGetObjectHandle(clientID, 'r_elbow_y', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, elbowY, math.radians(angle), vrep.simx_opmode_blocking)
    
def setWristZ(angle):
    [res3,wristZ]=vrep.simxGetObjectHandle(clientID, 'r_wrist_z', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, wristZ, math.radians(angle), vrep.simx_opmode_blocking)
    
def setWristX(angle):
    [res3,wristX]=vrep.simxGetObjectHandle(clientID, 'r_wrist_x', vrep.simx_opmode_blocking)
    vrep.simxSetJointTargetPosition(clientID, wristX, math.radians(angle), vrep.simx_opmode_blocking)
    
#--------------------------get--------------------------------
    
def getArm(): 
    [res3,armX]=vrep.simxGetObjectHandle(clientID, 'r_arm_x', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, armX, vrep.simx_opmode_blocking)[1])

def getShoulderZ(): 
    [res3,shoulderZ]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_z', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, shoulderZ, vrep.simx_opmode_blocking)[1])

def getShoulderY(): 
    [res3,shoulderY]=vrep.simxGetObjectHandle(clientID, 'r_shoulder_y', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, shoulderY, vrep.simx_opmode_blocking)[1])

def getElbow():
    [res3,elbowY]=vrep.simxGetObjectHandle(clientID, 'r_elbow_y', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, elbowY, vrep.simx_opmode_blocking)[1])
    
def getWristZ(): 
    [res3,wristZ]=vrep.simxGetObjectHandle(clientID, 'r_wrist_z', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, wristZ, vrep.simx_opmode_blocking)[1])
    
def getWristX(): 
    [res3,wristX]=vrep.simxGetObjectHandle(clientID, 'r_wrist_x', vrep.simx_opmode_blocking)
    return math.degrees(vrep.simxGetJointPosition(clientID, wristX, vrep.simx_opmode_blocking)[1])

