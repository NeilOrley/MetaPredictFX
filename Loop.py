import time
import subprocess

def execute_program():
    
    subprocess.run(['python', 'DeepLearningForecast.py'])

# Stablish and interval in seconds (for example, 60 seconds)
seconds_interval = 5


while True:
    execute_program()
    print(f"Program executed. Waiting {seconds_interval} seconds")

    # waits the interval before executing the program
    time.sleep(seconds_interval)
