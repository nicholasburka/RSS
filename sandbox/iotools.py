#!/usr/bin/env python
__IOTOOLS_VERSION__="1.0.0"

from Phidgets.PhidgetException import PhidgetErrorCodes, PhidgetException
from Phidgets.Events.Events import AttachEventArgs, DetachEventArgs, ErrorEventArgs, InputChangeEventArgs, OutputChangeEventArgs, SensorChangeEventArgs
from Phidgets.Devices.InterfaceKit import InterfaceKit
from Phidgets.Devices.MotorControl import MotorControl
from Phidgets.Devices.AdvancedServo import AdvancedServo
from Phidgets.Devices.Servo import ServoTypes
import threading
import time
import math
import cv2
import numpy
import os

class IOTools:
    def __init__(self, onRobot):
        self.onRobot = onRobot

        # Camera
        self._openCam=False
        
        # Motor Control
        self._snMot=-1
        self._openMot=False
        self._attachedMot=False
        self._cur = [0, 0]

        # Servo
        self._snSer=-1
        self._openSer=False
        self._attachedSer=False
        self._limits = [0, 0]
        
        # IF Kit
        self._snIF=-1
        self._openIF=False
        self._attachedIF=False        
        self._inp = [0, 0, 0, 0, 0, 0, 0, 0]
        self._sen = [0, 0, 0, 0, 0, 0, 0, 0]
        
        # LEDs
        self._fistTime = True
        self._status = [0,0,0]
        self._mod = [8, 1, 1]
        self._rep = [-1, 5, 6]
        self._val = [True, False, False]
        self._ofs = [0, 0, 0]
        self._updaterThread = threading.Thread(target=self.__updateLED)
        self._stop = False
        self._updaterThread.setDaemon(True)
        self._updaterThread.start()

    def destroy(self):
        # LEDs
        self._stop = True
        if self._attachedIF:
            self._interfaceKit.setOutputState(0, 0)
            self._interfaceKit.setOutputState(1, 0)
            self._interfaceKit.setOutputState(2, 0)
        # Servo
        self.servoDisengage()
        # Camera
        self._openCam = False
        if self._openCam:
            self._cap.release()

    def open(self):
        self.__openIF()
        self.__openMot()
        self.__openSer()
        self.__openCam()

###################### Camera ######################
        
    def __openCam(self):
        if not os.path.exists('/dev/video0'):
            return False
        self._cap = cv2.VideoCapture()
        if not self._cap.open(-1):
            return False
        self._openCam = True

    def cameraGrab(self):
        if self._openCam:
            return self._cap.grab()
        else:
            return False

    def cameraRead(self):
        if self._openCam:
            (ret, img)=self._cap.retrieve()
            return img
        else:
            return False

    def cameraSetResolution(self, sz):
        if self._openCam:
            sz=sz.lower()
            if sz=='low':
                self._cap.set(3,160)
                self._cap.set(4,120)
            if sz=='medium':
                self._cap.set(3,640)
                self._cap.set(4,480)
            if sz=='high':
                self._cap.set(3,800)
                self._cap.set(4,600)
            if sz=='full':
                self._cap.set(3,1280)
                self._cap.set(4,720)

    def imshow(self, wnd, img):
        if not self.onRobot:
            if img.__class__ != numpy.ndarray:
                print "imshow - invalid image"
                return False
            else:
                cv2.imshow(wnd,img)
                cv2.waitKey(5)
        
####################### Servo ######################

    def __openSer(self):
        try:
            self._advancedServo = AdvancedServo()
        except RuntimeError as e:
            print("Servo - Runtime Exception: %s" % e.details)
            return False        

        try:
            self._advancedServo.setOnAttachHandler(self.__onAttachedSer)
            self._advancedServo.setOnDetachHandler(self.__onDetachedSer)
            self._advancedServo.setOnErrorhandler(self.__onErrorSer)
        except PhidgetException as e:
            print("Servo - Phidget Exception %i: %s" % (e.code, e.details))
            return False
        
        try:
            self._advancedServo.openPhidget()
        except PhidgetException as e:
            print("Servo - Phidget Exception %i: %s" % (e.code, e.details))
            return False
        self._openSer=True
        return True

    def __onAttachedSer(self,e):
        self._snSer = e.device.getSerialNum()
        self._advancedServo.setServoType(0, ServoTypes.PHIDGET_SERVO_HITEC_HS322HD)
        self._advancedServo.setAcceleration(0, self._advancedServo.getAccelerationMax(0))
        self._advancedServo.setVelocityLimit(0, self._advancedServo.getVelocityMax(0))
        self._limits[0] = self._advancedServo.getPositionMin(0)
        self._limits[1] = self._advancedServo.getPositionMax(0)
        print("Servo %i Attached! Range: %f - %f" % (self._snSer, self._limits[0], self._limits[1]))
        self._attachedSer=True

    def __onDetachedSer(self,e ):
        print("Servo %i Detached!" % (self._snSer))
        self._snSer = -1
        self._attachedSer=False
              
    def __onErrorSer(self, e):
        try:
            source = e.device
            print("Servo %i: Phidget Error %i: %s" % (source.getSerialNum(), e.eCode, e.description))
        except PhidgetException as e:
            print("Servo - Phidget Exception %i: %s" % (e.code, e.details))

    def __closeSer(self):
        if self._openSer==True:
            self.servoDisengage()
            self._advancedServo.closePhidget()

    def servoEngage(self):
        if self._attachedSer==True:
            self._advancedServo.setEngaged(0, True)

    def servoDisengage(self):
        if self._attachedSer==True:
            self._advancedServo.setEngaged(0, False)

    def servoSet(self, pos):
        if self._attachedSer==True:
            self._advancedServo.setPosition(0, min(max(pos,self._limits[0]),self._limits[1]))
        
############### Motor Control ######################

    def __openMot(self):
        try:
            self._motorControl = MotorControl()
        except RuntimeError as e:
            print("Motor Control - Runtime Exception: %s" % e.details)
            return False        

        try:
            self._motorControl.setOnAttachHandler(self.__onAttachedMot)
            self._motorControl.setOnDetachHandler(self.__onDetachedMot)
            self._motorControl.setOnErrorhandler(self.__onErrorMot)
            self._motorControl.setOnCurrentChangeHandler(self.__onCurrentChangedMot)
        except PhidgetException as e:
            print("Motor Control - Phidget Exception %i: %s" % (e.code, e.details))
            return False
        
        try:
            self._motorControl.openPhidget()
        except PhidgetException as e:
            print("Motor Control - Phidget Exception %i: %s" % (e.code, e.details))
            return False
        self._openMot=True
        return True

    def __onAttachedMot(self,e):
        self._snMot = e.device.getSerialNum()
        print("Motor Control %i Attached!" % (self._snMot))
        self._attachedMot=True

    def __onDetachedMot(self,e ):
        print("Motor Control %i Detached!" % (self._snMot))
        self._snMot = -1
        self._attachedMot=False
              
    def __onErrorMot(self, e):
        try:
            source = e.device
            print("Motor Control %i: Phidget Error %i: %s" % (source.getSerialNum(), e.eCode, e.description))
        except PhidgetException as e:
            print("Motor Control - Phidget Exception %i: %s" % (e.code, e.details))

    def __onCurrentChangedMot(self, e):
        self._cur[e.index] = e.current

    def __closeMot(self):
        if self._openMot==True:
            self._motorControl.closePhidget()

    def setMotors(self, speed1=0.0, speed2=0.0, acceleration1=100.0, acceleration2=100.0 ):
        if self._openMot==True:
            self._motorControl.setAcceleration(0, acceleration1)
            self._motorControl.setVelocity(0, speed1)
            self._motorControl.setAcceleration(1, acceleration2)
            self._motorControl.setVelocity(1, speed2)
            
############### Interface Kit ######################

    def __closeIF(self):
        if self._openIF==True:
            self._interfaceKit.closePhidget()
            
    def __openIF(self):
        try:
            self._interfaceKit = InterfaceKit()
        except RuntimeError as e:
            print("IF Kit - Runtime Exception: %s" % e.details)
            return False
        try:
            self._interfaceKit.setOnAttachHandler(self.__onAttachedIF)
            self._interfaceKit.setOnDetachHandler(self.__onDetachedIF)
            self._interfaceKit.setOnErrorhandler(self.__onErrorIF)
            self._interfaceKit.setOnInputChangeHandler(self.__onInputChangedIF)
            self._interfaceKit.setOnSensorChangeHandler(self.__onSensorChangedIF)
        except PhidgetException as e:
            print("IF Kit - Phidget Exception %i: %s" % (e.code, e.details))
            return False
        try:
            self._interfaceKit.openPhidget()
        except PhidgetException as e:
            print("IF Kit - Phidget Exception %i: %s" % (e.code, e.details))
            return False
        self._openIF=True
        return True
        
    def __onAttachedIF(self,e):
        self._snIF = e.device.getSerialNum()
        print("InterfaceKit %i Attached!" % (self._snIF))
        self._attachedIF=True
        if self._fistTime:
            for i in range(0,3):
                self._interfaceKit.setOutputState(0, 1)
                self._interfaceKit.setOutputState(1, 1)
                self._interfaceKit.setOutputState(2, 1)
                time.sleep(0.1)
                self._interfaceKit.setOutputState(0, 0)
                self._interfaceKit.setOutputState(1, 0)
                self._interfaceKit.setOutputState(2, 0)
                time.sleep(0.1)
        self._fistTime = False

    def __onDetachedIF(self,e ):
        print("InterfaceKit %i Detached!" % (self._snIF))
        self._snIF = -1
        self._inp = [0, 0, 0, 0, 0, 0, 0, 0]
        self._sen = [0, 0, 0, 0, 0, 0, 0, 0]
        self._attachedIF=False
              
    def __onErrorIF(self, e):
        try:
            source = e.device
            print("InterfaceKit %i: Phidget Error %i: %s" % (source.getSerialNum(), e.eCode, e.description))
        except PhidgetException as e:
            print("IF Kit - Phidget Exception %i: %s" % (e.code, e.details))

    def __onSensorChangedIF(self, e):
        self._sen[e.index] = e.value

    def __onInputChangedIF(self, e):
        self._inp[e.index] = e.state

    def getSensors(self):
        return self._sen;

    def getInputs(self):
        return self._inp;

################ LEDs #######################

    def __updateLED(self):
        t=0
        while self._stop==False:
            t=(t+1)%100
            for i in range(0,3):
                self._status[i]=((t+self._ofs[i])%self._mod[i]==0) and self._val[i] and bool(self._rep[i])
                self._rep[i]=self._rep[i]-int(self._rep[i]>0 and self._status[i])
                if self._attachedIF:
                    self._interfaceKit.setOutputState(i, self._status[i])
            time.sleep(0.15)
            
    def __setModeLED(self, i, mode, hz=2, cnt=1, ofs=0):
        if mode=='on':
            self._rep[i]=-1
            self._val[i]=True
            self._ofs[i]=0
            self._mod[i]=1
        if mode=='off':
            self._rep[i]=-1
            self._val[i]=False
            self._ofs[i]=0
            self._mod[i]=1
        if mode=='flash':
            hz=min(max(hz,1),100)
            self._rep[i]=min(max(cnt,1),20)
            self._val[i]=True
            self._ofs[i]=min(max(ofs,0),hz)
            self._mod[i]=hz

    def setStatus(self, mode, hz=2, cnt=1, ofs=0):
        self.__setModeLED(1,mode, hz, cnt, ofs)            
                
    def setError(self, mode, hz=2, cnt=1, ofs=0):
        self.__setModeLED(2,mode, hz, cnt, ofs)

    def setSemaphor(self):
        self.__setModeLED(1,'flash', 2, 6, 0)
        self.__setModeLED(2,'flash', 2, 6, 1)
