#!/usr/bin/env python
__SNADBOX_VERSION__="1.0.0"

import iotools
import traceback
import datetime
import threading
import time
import signal
import os
import sys

class Logger(object):
    def __init__(self, onRobot):
        self.useTerminal=not onRobot
        self.terminal = sys.stdout
        self.log = 0
        if os.path.isdir('/tmp/sandbox'):
            self.log = open("/tmp/sandbox/log.txt", "a+")

    def write(self, message):
        if self.useTerminal:
            self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()

class SandBox:
    def __init__(self, onRobot):
        self._version = "R:SS SandBox "+__SNADBOX_VERSION__
        print datetime.datetime.now()
        self.version()
        self._IO=iotools.IOTools(onRobot)
        try:
            self._IO.open()
            time.sleep(2.0)
            sys.path.insert(0,'/home/student/')
            sys.path.insert(1,'/home/tc/') 
            import toddler
            self._toddler = toddler.Toddler(self._IO)
            self._wrk = Workers(self._IO, self._toddler.Control, self._toddler.Vision)
        except Exception:
            self._IO._mod[0]=2
            print("The toddler is hurt:")
            traceback.print_exc()     

    def version(self):
        print self._version
        return self._version

    def destroy(self):
        self._wrk.destroy()
        self._IO.destroy()

class Workers:
    def __init__(self, IO, ctrl, vis):
        self._IO = IO
        self._OK = True
        self._workers = [[ctrl, 0, 1], [vis, 0 , 1], [self.__led, 0, 1]]
        for i in range(0,len(self._workers)):
            self._workers[i][1] = threading.Thread(target=self.__workerThread,args=[self._workers[i][0],i])
            self._workers[i][1].setDaemon(True)
            self._workers[i][1].start()

    def __isOK(self):
        return self._OK

    def __workerThread(self, callback, i):
        while self._OK:
            try:
                callback(self.__isOK)
                self._workers[i][2]=1
            except Exception:
                self._workers[i][2]=0
                print("Callback failed:")
                traceback.print_exc()                

    def __led(self, OK):
        st=0
        for wrk in self._workers:
            st+=wrk[2]
        if st==len(self._workers):
            self._IO._mod[0]=8
        else:
            self._IO._mod[0]=2
        time.sleep(0.2)

    def wait(self):
        for wrk in self._workers:
            wrk[1].join()
            
    def destroy(self):
        self._OK = False

def sigterm_handler(_signo, _stack_frame):
    sys.exit(0)

if __name__ == "__main__":
    signal.signal(signal.SIGTERM, sigterm_handler)
    try:
        onRobot=bool(sys.argv.count('-rss'))
        sys.stdout = Logger(onRobot)
        sys.stderr = sys.stdout
        sb = SandBox(onRobot)
        while threading.active_count() > 0:
            time.sleep(0.1)
    except (KeyboardInterrupt, SystemExit):
        sb.destroy();
        
