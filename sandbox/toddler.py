#!/usr/bin/env python
__TODDLER_VERSION__="1.0.0"

import time
import numpy
import cv2
import datetime

# Hardware test code
class Toddler:
    def __init__(self,IO):
        print 'I am a toddler playing in a sandbox'
        self.IO=IO
        self.inp=[0, 0, 0, 0, 0, 0, 0, 0]
        self.TEST = False

    def move(self, l,r):
      if not l and not r:
        return [0, 0]
      if l and not r:
        return [100, -100]
      if not l and r:
        return [-100, 100]
      if l and r:
        return [100, 100]    
        

    # This is a callback that will be called repeatedly.
    # It has its dedicated thread so you can keep block it.
    def Control(self, OK):
        mot=[False,False,False]
        motPrev=[False,False,False]
        pos=180
        while OK():
            time.sleep(0.05)
            self.inp=self.IO.getInputs()
            mot[0]=self.inp[1]
            mot[1]=self.inp[2]
            mot[2]=self.inp[3]
            if mot[0]!=motPrev[0] or mot[1]!=motPrev[1]:
                speed = self.move(mot[0],mot[1])
                self.IO.setMotors(-speed[0],speed[1])
                if mot[0]:
                    self.IO.setStatus('on')
                else:
                    self.IO.setStatus('off')
                if mot[1]:
                    self.IO.setError('on')
                else:
                    self.IO.setError('off')
                
                    
            if mot[2]:
                self.IO.servoEngage()
                pos=(pos+3)%360
                self.IO.servoSet(abs(pos-180))
            if mot[2]!=motPrev[2] and not mot[2]:
               self.IO.servoDisengage()
            motPrev[0]=mot[0]
            motPrev[1]=mot[1]
            motPrev[2]=mot[2]

    # This is a callback that will be called repeatedly.
    # It has its dedicated thread so you can keep block it.
    def Vision(self, OK):
        self.IO.cameraSetResolution('low')
        hasImage=False
        res=0
        sw=False
        swPrev=False
        while OK():
            if self.inp[4]:
                # Why is this in range 0-5?
                for i in range(0,5):
                    self.IO.cameraGrab()
                img=self.IO.cameraRead()
                if img.__class__==numpy.ndarray:
                    hasImage=True
                    # save the image to a file
                    if self.TEST:
                        cv2.imwrite('camera-'+datetime.datetime.now().isoformat()+'.png',img)
                    self.IO.imshow('window',img)
                    self.IO.setStatus('flash',cnt=2)
                    time.sleep(0.5)
            if hasImage:    
                # redundant
                self.IO.imshow('window',img)
                
            sw=self.inp[5]
            if sw!=swPrev and sw:
                res=(res+1)%4
                if res==0:
                    self.IO.cameraSetResolution('low')
                    self.IO.setError('flash',cnt=1)
                if res==1:
                    self.IO.cameraSetResolution('medium')
                    self.IO.setError('flash',cnt=2)
                if res==2:
                    self.IO.cameraSetResolution('high')
                    self.IO.setError('flash',cnt=3)
                if res==3:
                    self.IO.cameraSetResolution('full')
                    self.IO.setError('flash',cnt=4)
                time.sleep(0.5)
            swPrev=sw
            
            time.sleep(0.05)
            

        
