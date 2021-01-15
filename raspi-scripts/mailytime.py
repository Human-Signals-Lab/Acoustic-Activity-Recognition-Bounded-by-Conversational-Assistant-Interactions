from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import smtplib
import socket
from email.mime.text import MIMEText
import datetime
from time import sleep
import checkRunningProcess
import re
from uuid import getnode as get_mac
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders

def mailyStatus():
    global count_hour
    to = 'your-gmail@gmail.com'
    gmail_user = 'your-gmail@gmail.com'
    gmail_password = 'YOURPASSWORD'
    smtpserver = smtplib.SMTP('smtp.gmail.com',587)
    smtpserver.ehlo()
    smtpserver.starttls()
    smtpserver.login(gmail_user,gmail_password)
    today = datetime.date.today()

    mac = get_mac()
    msg = MIMEMultipart()
    mac_p = ':'.join(("%012X" % mac)[i:i+2] for i in range(0,12,2))
    process_name = b"/home/pi/lightSense.py"
    result = checkRunningProcess.isThisRunning(process_name)
    if result:
        msg.attach(MIMEText("LightSense script is running."))
        msg['Subject'] = str(mac_p) + ': RPi3 LightSense Status Update: Running smoothly!'
    else:
        msg.attach(MIMEText("LightSense script is not running."))
        msg['Subject'] = str(mac_p) + ': RPi3 LightSense Status Update: ACTION NEEDED!'

    scriptLog = "/home/pi/lightSense_log.log"
    
    with open(scriptLog,"rb") as attachment:
        part = MIMEBase("application","octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition","attachment", filename= "python_script.log")
    msg.attach(part)
    
    
    shLog = "/home/pi/light_sensing_log.log"
    
    with open(shLog,"rb") as attachment:
        part = MIMEBase("application","octet-stream")
        part.set_payload(attachment.read())
    encoders.encode_base64(part)
    part.add_header("Content-Disposition","attachment", filename= "shell_script.log")
    msg.attach(part)
    
    
    msg['From'] = gmail_user
    msg['To'] = to
    smtpserver.sendmail(gmail_user, [to], msg.as_string())
    smtpserver.quit()
 
scheduler = BackgroundScheduler()
scheduler.add_job(mailyStatus, 'interval', minutes = 60)
scheduler.start()
while True:
    continue