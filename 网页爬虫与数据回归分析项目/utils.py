import re
def get_incomplete_data(Info,type):
    if type is 'decoration':
        if len(Info) == 5:
            return Info[4].strip(' ')
        else:
            return ' '
    elif type is 'distance':
        if Info is ' ':
            return Info
        else:
            return re.findall(r"\d+", Info)

