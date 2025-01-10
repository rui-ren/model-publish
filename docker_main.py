import subprocess
import time
import sys
from evaluate_models_target_ip import async_main
from evaluate_models_target_eci import async_main_eci
from evaluate_models_target_jailbreak import async_main_jailbreak
from evaluate_models_target_ground import async_main_ground

import os
print("### ENV in docker_main.py")
print(os.environ)

# step.1 - launch "uvicorn main:app --reload" in shell
uvicorn_proc = subprocess.Popen(["uvicorn main:app --reload"], shell=True)

# step.2 - wait for the server ready
time.sleep(20)

# step.3 - call the endpoint and do validation
print(" ### Start model endpoint :)")

import asyncio
loop = asyncio.get_event_loop()
loop.run_until_complete(async_main())
loop.run_until_complete(async_main_eci())
loop.run_until_complete(async_main_jailbreak())
loop.run_until_complete(async_main_ground())



# step.last - send SIGTERM to server and wait for it to exit
uvicorn_proc.send_signal(subprocess.signal.SIGTERM)
uvicorn_proc.wait()

# run baseline model with RAI evaluation
# uvicorn_proc = subprocess.Popen(["uvicorn main_pytorch:app --reload"], shell=True)


# time.sleep(25)


# print(" ### Start baseline-model endpoint :)")

# import asyncio
# loop = asyncio.get_event_loop()
# loop.run_until_complete(async_main())
# loop.run_until_complete(async_main_eci())
# loop.run_until_complete(async_main_jailbreak())
# loop.run_until_complete(async_main_ground())



# uvicorn_proc.send_signal(subprocess.signal.SIGTERM)
# uvicorn_proc.wait()

