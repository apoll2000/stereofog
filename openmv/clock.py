import RPi.GPIO as GPIO
import time

# References:
# https://codeandlife.com/2012/07/03/benchmarking-raspberry-pi-gpio-speed/
# https://raspberrypi.stackexchange.com/questions/55143/gpio-warning-channel-already-in-use

GPIO.setmode(GPIO.BCM)

GPIO.setup(4, GPIO.OUT)

try:
    while True:
        GPIO.output(4, True)
        time.sleep(0.5)
        GPIO.output(4, False)
        time.sleep(2)
    
except KeyboardInterrupt: # If CTRL+C is pressed, exit cleanly:
   print("Keyboard interrupt")

except:
   print("some error") 

finally:
   print("clean up") 
   GPIO.cleanup() # cleanup all GPIO 