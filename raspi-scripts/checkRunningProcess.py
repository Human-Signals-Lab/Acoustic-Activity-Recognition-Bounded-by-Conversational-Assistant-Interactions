import os
import psutil
from subprocess import PIPE
import re

def findThisProcess(process_name):
    ps = psutil.Popen(b"ps -ef | grep " + process_name + b"| grep -v grep", shell=True, stdout = PIPE)
    output = ps.stdout.read()
    print(output)
    ps.stdout.close()
    ps.wait()
    
    return output

def isThisRunning(process_name):
    output = findThisProcess(process_name)
    print(process_name)
    if re.search(process_name, output) is None:
        return False
    else:
        return True

if __name__ == "__main__":
    process_name = b"./lightSense"
    print(psutil.test())
    if isThisRunning(process_name) == False:
        print("Not Running")
    else:
        print("Running")
        