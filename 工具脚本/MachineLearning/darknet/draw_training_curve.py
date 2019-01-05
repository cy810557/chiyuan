import os
import sys
import re
import time
import pdb
import atexit
from datetime import datetime
import matplotlib.pyplot as plt
from subprocess import *

if len(sys.argv)>1:
    curve_start = int(sys.argv[1])
else:
    curve_start = 0

smooth_step = int(sys.argv[2])

def exit_handler():
    plt.savefig('training_curve.png')
atexit.register(exit_handler)

plt.ion()
plt.figure(figsize=(20,20))
while(True):
    print(datetime.now().strftime("%H-%M-%S"))
    Popen("cat ../../tiny-yolo-train-1024-1024-phase-2.log | grep images | grep : > tiny-yolo-train-1024-1024-step-phase-2.log",shell=True)
    
    #log_path = 'tiny-yolo-train-phase2-step.log'
    #time.sleep(0.05)
    #total_loss_content = Popen("cat tiny-yolo-train-1024-1024-step.log | cut -f2 -d ' ' ", stdout = PIPE,shell=True).stdout.read().decode("utf-8").split(',\n') ## decode: bytes to string

    #total_loss = [float(x) for x in total_loss_content[curve_start:-1]]
    #pdb.set_trace()
    #time.sleep(0.05)
    with open('tiny-yolo-train-1024-1024-step-phase-2.log', 'r') as f:
        content = f.read()
    total_loss_content = re.findall(": (.*?), ",content)
    avg_loss_content = re.findall(", (.*) avg",content)
    lr = re.findall("avg, (.*?) rate",content)
    total_loss = [float(x) for x in total_loss_content[curve_start:]]
    avg_loss = [float(x) for x in avg_loss_content[curve_start:]]
    #avg_loss_content = Popen("cat tiny-yolo-train-1024-1024-step.log | cut -f3 -d ' ' ", stdout = PIPE,shell=True).stdout.read().decode("utf-8").split('\n')
    #avg_loss = [float(x) for x in avg_loss_content[curve_start:-1]]
     
    steps = list(range(len(avg_loss)))
    
    plt.subplot(1,2,1)
    #plt.plot(steps[::smooth_step], lr[::smooth_step], 'g-.')
    plt.plot(steps[::smooth_step], total_loss[::smooth_step], 'r-')
    plt.legend(labels = ['total loss'], loc = 'upper right') #'learning rate',
    plt.subplot(1,2,2)
    #plt.plot(steps[::smooth_step], lr[::smooth_step], 'g-.')
    plt.plot(steps[::smooth_step], avg_loss[::smooth_step], 'b-')

    plt.legend(labels = ['avg loss'], loc = 'upper right')
    plt.show();
    plt.pause(157)
    plt.clf()
plt.savefig('training_curve.png')
