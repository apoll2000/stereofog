# read_external_pin - By: Anton Pollak - Sat Jun 24 2023
# this script reads a high/low signal from a GPIO pin and takes a snapshot at a high value
# GPIO tutorial: https://docs.openmv.io/openmvcam/tutorial/gpio_control.html

import sensor, image, time, pyb, time, mjpeg
import os

if 'recorded_images' not in os.listdir():
    os.mkdir('recorded_images')
if 'recorded_videos' not in os.listdir():
    os.mkdir('recorded_videos')

sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.skip_frames(time = 2000) # to let the camera stabilize after changing settings

clock = time.clock()

counter = len(os.listdir('recorded_images')) + 1
video_counter = len(os.listdir('recorded_videos')) + 1

#print('counter:', counter)

pin = pyb.Pin("P0", pyb.Pin.IN)
mode_pin = pyb.Pin("P1", pyb.Pin.IN, pyb.Pin.PULL_UP)

# Initialize the LED
green_led = pyb.LED(2)
blue_led = pyb.LED(3)
green_led.off()
blue_led.off()


# Loop forever
while(True):
    # print(pin.value()) # print GPIO value to serial terminal

    if mode_pin.value() == 1:

        blue_led.on()

        m = mjpeg.Mjpeg("recorded_videos/"+ str(video_counter) + ".mjpeg")

        while mode_pin.value() == 1:
            clock.tick()
            m.add_frame(sensor.snapshot())
            #print(clock.fps())

        m.close(clock.fps())
        blue_led.off()

        video_counter += 1

        #while mode_pin.value() == 1:
            #pass # do nothing to wait for the trigger going away, to make sure only one image is collected per trigger

    else:
        # collect image if GPIO pin detects a HIGH signal
        if pin.value() == 1:


                img = sensor.snapshot()

                # toggle green LED after recording image to provide positive user feedback
                green_led.on()
                time.sleep_ms(100)
                green_led.off()

                # Saving the image
                img.save('/recorded_images/' + str(counter))
                counter += 1

                # Stop continuing until the pin value has gone to low again
                while pin.value() == 1:
                    pass # do nothing to wait for the trigger going away, to make sure only one image is collected per trigger


        else:
            # Do nothing
            pass
