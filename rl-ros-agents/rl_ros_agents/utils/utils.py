from datetime import datetime

def getTimeStr():
    time = datetime.now()
    return time.strftime("%Y_%m_%d_%H_%M")
