# This file runs on the servers

from jtop import jtop
from datetime import datetime
import time 

with jtop() as jetson:
    # jetson.ok() will provide the proper update frequency
    while jetson.ok():
        start_time = datetime.now()
        
        # Read power stats
        timestamp = jetson.stats['time']
        # Orin: power = jetson.stats['Power VIN_SYS_5V0']
        power = jetson.stats['Power TOT']
        
        print(timestamp, ": ", power)
        
        # Sleep for remaining time to complete 1 second interval
        elapsed = (datetime.now() - start_time).total_seconds()
        time.sleep(max(0, 1 - elapsed))