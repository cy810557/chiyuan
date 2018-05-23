#!/usr/bin/env python
import os
import sys
if(len(sys.argv)<3):
    print"please input 2 arguments"
arg0 = sys.argv[1]
arg1 = sys.argv[2]
os.system('./hello.sh ' +arg0+' '+arg1)
