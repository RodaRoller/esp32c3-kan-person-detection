#include <Arduino.h>
#include <esp_arduino_version.h>

void setup() {
    Serial.begin(115200);
}

void loop() {
    Serial.printf("C3-03 SERIAL READY | %s | %lu MHz | flash=%lu B | "
                  "Arduino=%d.%d.%d | IDF=%s | GCC=%s\n",
                  ESP.getChipModel(),
                  static_cast<unsigned long>(ESP.getCpuFreqMHz()),
                  static_cast<unsigned long>(ESP.getFlashChipSize()),
                  ESP_ARDUINO_VERSION_MAJOR, ESP_ARDUINO_VERSION_MINOR,
                  ESP_ARDUINO_VERSION_PATCH, ESP.getSdkVersion(), __VERSION__);
    delay(2000);
}
