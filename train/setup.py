import subprocess
import sys
import os

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package, "--user"])

install("OpenCV-python")
install("tensorflow")
install("scipy")