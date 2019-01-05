import os
import pyperclip
import time
copyBuff=' '
while True:
    time.sleep(10)
    copyedText=pyperclip.paste()
    if copyBuff!=copyedText:  #check whether clipboard changed or not
        copyBuff=copyedText
        normalizedText = copyBuff.replace(os.linesep, ' ') 
        #linespe return '\r\n' for windows;'\n' for linux and '\r' for mac
        pyperclip.copy(normalizedText)
    else:
        print('no change')