/*
# References: https://stackoverflow.com/questions/45320181/adafruit-feather-huzzah-esp8266-pull-up-resistor-using-arduino-ide
# https://learn.adafruit.com/adafruit-feather-32u4-radio-with-rfm69hcw-module/using-with-arduino-ide
# https://www.deviceplus.com/arduino/the-basics-of-arduino-reading-switch-states/
*/

const int TOGGLE_PIN = 18;
const int TRIGGER_SWITCH = 5;

void setup() {
  pinMode(TOGGLE_PIN, OUTPUT);
  pinMode(TRIGGER_SWITCH, INPUT_PULLUP);
  pinMode(LED_BUILTIN, OUTPUT);

  delay(4000);
}

void loop() {

  int trigger_switch_value;

  trigger_switch_value = digitalRead(TRIGGER_SWITCH);


  if (trigger_switch_value == LOW) {
    digitalWrite(LED_BUILTIN, HIGH);
    digitalWrite(TOGGLE_PIN, HIGH);
    delay(500);
    digitalWrite(TOGGLE_PIN, LOW);
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
  }
}