#!/usr/bin/python
# coding=utf-8
import os
import argparse
import time
import pdb
from pyvirtualdisplay import Display
from selenium import webdriver
from selenium.webdriver.chrome.service import Service


args = argparse.ArgumentParser()
args.add_argument("-u", "--user", default="121101701064100",
                  help="Username of your njupt account.")
args.add_argument("-k", "--key", default="xxxxxx",
                  help="Keyword to login njupt.")
args.add_argument("-m", "--mode", choices=["login", "logout"], default="login",
                  help="Mode: login or logout. Default: login") 
args = args.parse_args()

def is_connected():
    #pdb.set_trace()
    ip = "www.baidu.com"
    backinfo = os.system('ping -c 1 -w 1 %s'%ip)  # 1 when ping can access ip.
    print("backinfo: {}".format(backinfo))
    if backinfo == 256:
        print("[INFO] Connection not established.")
        return False
    elif backinfo == 0:
        print("[INFO] Connection has already been established! ")
        return True
    else:
        print("[INFO] Unexpected backinfo encoutered! Expected 0 or 256, received {}".format(backinfo))
        exit(0)


if __name__ == "__main__":
    mode = args.mode
    display = Display(visible=0)
    display.start()
    #c_service = Service('[driver_location]')
    #c_service.start()

    driver = webdriver.Chrome()
    driver.get("http://192.168.168.168")
    time.sleep(1)
    connected = is_connected()
    if connected and mode == "login":
        exit(0) 
    elif not connected and mode == "logout":
        exit(0)
    elif connected and mode == "logout":
        elem_logout = driver.find_element_by_id("submit")
        elem_logout.click()
        time.sleep(2)
        elem_alert = driver.switch_to.alert 
        time.sleep(1)
        elem_alert.accept()
        time.sleep(1)
        print("[INFO] Loged out successfully!")
    else:
        print("[INFO] Building connection ...") 
        elem_username = driver.find_element_by_name("DDDDD")
        elem_username.send_keys(args.user)

        elem_key = driver.find_element_by_name("upass") 
        elem_key.send_keys(args.key)
        elem=driver.find_element_by_id('submit').click()
        print("successfully loged to NJUPT")
        
    driver.quit()
    display.stop()
    #c_service.stop()


