# stereofog
This repository documents a research project carried out at the [Laboratory for Optical Nanotechnologies](https://nanoptics.wordpress.com) at the University of Utah under supervision of [Prof. Rajesh Menon](https://faculty.utah.edu/u0676529-Rajesh_Menon/research/index.hml) in Summer 2023 (July-September). It was funded as part of the [RISE program](https://www.daad.de/rise/en/) by the [German Academic Exchange Service (Deutscher Akademischer Austauschdienst (DAAD))](https://en.wikipedia.org/wiki/German_Academic_Exchange_Service).


*insert preliminary results, logos, images, etc. as teaser here*

## Goal
This project had three objectives:

1. build a device capable of capturing paired images that depict the same scenery, one image with fog and the other without
2. collect a dataset of paired images
3. apply the [pix2pix model](https://phillipi.github.io/pix2pix/) developed at the University of California, Berkeley to the translation problem **fog &rarr; no fog**

## Project timeline
![project timeline](images/gantt_chart_stereofog.png "project timeline")

## Image capturing device
### Requirements

The device had to be able to:
* accomodate two cameras
* isolate the cameras from each other
* provide a fog chamber for one of the cameras
* trigger both cameras at the same time

The shift in perspective hailing from the distance the two cameras are set apart will be ignored. The further away the photographed scenes are, the less this will have an effect on the resulting images.

### Cameras
The two identical cameras used for this project had to be:

* programmable
* able to interface with other devices
* small & lightweight
* low power

Therefore, we chose to use the [OpenMV](https://openmv.io) [H7](https://openmv.io/collections/products/products/openmv-cam-h7) cameras for the task. The [OpenMV IDE](https://openmv.io/pages/download) makes it easy to program the camera using `python`. They are able to receive input from their I/O pins as well as output user feedback using their LEDs.

<p align="center">
<img src="https://openmv.io/cdn/shop/products/new-cam-v4-angle-web_grande.jpg?v=1536445279" alt="OpenMV H7 camera" width="400"/>
</p>

### Image trigger
In order to get exactly paired images from both cameras that are captured at the same time, it is necessary to introduce a common trigger. We used a lightweight Arduino board for this task. Any Arduino board should be capable of sending this trigger, but we used an [Adafruit Feather 32u4 Radio](https://learn.adafruit.com/adafruit-feather-32u4-radio-with-rfm69hcw-module) that was available from an earlier project.

<p align="center">
<img src="https://cdn-learn.adafruit.com/guides/cropped_images/000/001/271/medium640/thumb.jpg?1520544029" alt="Adafruit Feather 32u4 Radio board" width="400"/>
</p>

The board is connected to both cameras and sends a trigger signal to both cameras at the same time. The cameras are programmed to capture an image when they receive the trigger signal.

### Wiring & Programming
<p align="center">
<img src="images/fog_device_schematics.png" alt="Schematics for the fog device" width="400"/>
</p>

**Image Attributions**:
[Switches](https://www.autobotic.com.my/Button-Switch/3-Pin-SPDT-Toggle-Switch-ON-OFF-ON) |
[Breadboard](https://magpi.raspberrypi.com/articles/breadboard-tutorial) |
[Adafruit Feather board](https://learn.adafruit.com/adafruit-feather-32u4-basic-proto/pinouts) |
[OpenMV camera](http://wiki.amperka.ru/_media/products:openmv-cam-h7:openmv-cam-h7-pinout.pdf)

Above is the wiring diagram for the project. Two switches are used to trigger both photos as well as videos. The photo trigger switch is connected to the Arduino board. It detects the state of the pin the switch is connected to and starts the recording loop. This means it sends a trigger signal to the cameras every second, as long as the switch is activated. At the same time, the onboard LED indicates this by blinking:

```arduino
  if (trigger_switch_value == LOW) {
    digitalWrite(LED_BUILTIN, HIGH);
    digitalWrite(TOGGLE_PIN, HIGH);
    delay(500);
    digitalWrite(TOGGLE_PIN, LOW);
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
  }
```

The entire Arduino code can be found in the [Arduino script](arduino/fog_device_trigger.ino).
In addition, the Arduino board is used to pass the 5V power supplied using a USB cable and a powerbank to the cameras.

The video trigger switch is connected directly to the cameras in order to avoid overhead introduced by the Arduino board.

Both OpenMV cameras are equipped with the exact same `python` code that listens to the two pins at which the input signals arrive. In case a video trigger signal is detected, the cameras instantly start recording a video. The video is stopped when the switch returns to the off position. The video is then saved to the microSD card as an `.mjpeg` file, numbered sequentially:

```python
# Loop forever
while(True):

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
```

In case no input signal is detected at the video trigger, the cameras listen to the photo trigger. When a signal is detected, they capture an image, label it sequentially, save it to the microSD card and then wait for the current trigger signal to go away, as to avoid capturing multiple images on one trigger:

```python
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
```

The entire `python` code for the cameras can be found in the respective [script](openmv/read_external_trigger.py).

### Gimbal

<p align="center">
<img src="https://www.bhphotovideo.com/cdn-cgi/image/format=auto,fit=scale-down,width=500,quality=95/https://www.bhphotovideo.com/images/images500x500/hohem_isteady_q_blk_isteady_q_smart_selfie_1661443536_1706042.jpg" alt="Schematics for the fog device" width="400"/>
</p>

In order to stabilize the images while walking and ensure they are approximately level, a gimbal was used to hold the entire device. The gimbal used for this project was the [Hohem iSteady Q](https://store.hohem.com/products/isteady-q). It is a lightweight single-axis gimbal that is able to hold a smartphone.

### Case

In order to be able to capture approximately the same image, ...

<p align="center">
<img src="images/gimbal_mount_3D_print.png" alt="CAD file of the gimbal mount" width="250"/>
</p>
<p align="center">
<img src="images/gimbal_bridge_3D_print.png" alt="CAD file of the gimbal mount" width="200"/>
</p>
<p align="center">
<img src="images/fog_device_assembly_3D_print_box_front.png" alt="CAD file of the gimbal mount" width="600"/>
</p>
<p align="center">
<img src="images/fog_device_assembly_3D_print_box_rear.png" alt="CAD file of the gimbal mount" width="600"/>
</p>

### 

## Python environment

Python version, requirements.txt, etc.

## Synthetic data
Uni Tübingen, Germany...

### Foggy Cityscapes from Uni Tübingen

### Foggy Cityscapes from ETH Zürich

[Link](http://people.ee.ethz.ch/~csakarid/SFSU_synthetic/)

### Foggy Carla from Uni Tübingen

### Foggy KITTI from Uni Tübingen (?)
tbd

## Collected dataset

*description & details of the collected dataset*
## pix2pix on dataset

*ML results on dataset*