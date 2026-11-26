# C3-03 · Porting della testa KAN su ESP32-C3 SuperMini

Estrai il contenuto di questa cartella in `c3-03/` nel tuo repository, accanto a `c3-02/`. Per i risultati di riferimento usa il tag `c3-02-r2` (commit `17db9a9c`). Il pacchetto dà la configurazione e il programma di confronto. In C3-03 svilupperai il generatore degli header e il motore C++.

## 1. Avvio sulla stessa piattaforma del laboratorio

La scheda fisica è la **ESP32-C3 SuperMini**, RISC-V RV32IMC, 160 MHz, 4 MB di flash, senza FPU. In PlatformIO il profilo corrispondente è `esp32-c3-devkitm-1`. Usiamo Arduino-ESP32 **2.0.17** (pacchetto `3.20017.241212`, ESP-IDF 4.4.7), Espressif32 **6.12.0**, GCC RISC-V **8.4.0+2021r2-patch5** ed esptool **4.9.0**. Le versioni dei pacchetti sono fissate in `platformio.ini`. Per questa procedura abbiamo scelto PlatformIO Core 6.1.19. [Scheda e foto](https://github.com/KuznetsovKarazin/lut-kan/blob/004139787a0fa623386ec071383b3b4ae350334f/README.md#L10-L14) · [Configurazione fisica del laboratorio](https://github.com/KuznetsovKarazin/kan-ids-v2/blob/81d99d9020d60b39d3b5a65fc6f06296f89537db/experiments/hardware_energy_20260915/firmware/c3/project/platformio.ini) · [Log di compilazione](https://github.com/KuznetsovKarazin/lut-kan-native/blob/49695234f54b986459b7932ce951fe4cafec6e37/experiments/sensor_recovery/results/matched_mcu_20261108/builds/esp32c3_matched_-1.log) · [Misure sulla SuperMini](https://github.com/KuznetsovKarazin/lut-kan-ids/blob/b0660d023f36c965d59511c759c34b143a3fb382/results/LUT_KAN_NODIV_K4_L8_hw.json)

Esegui questi comandi dalla cartella `c3-03/` in un ambiente Python attivo.

```sh
python -m pip install "platformio==6.1.19"
pio run -e c3_supermini
pio pkg list -e c3_supermini
```

Il programma iniziale scrive `C3-03 SERIAL READY`, modello del chip, clock, flash e versioni ogni due secondi. Nel progetto apri **Wokwi: Start Simulator** in VS Code, oppure apri un progetto ESP32-C3 su Wokwi.com, usa il `diagram.json` fornito e dal menu F1 scegli **Upload Firmware and Start Simulation** per il file pubblico F0. Per la prima prova puoi caricare il binario diagnostico. In VS Code `wokwi.toml` punta al binario prodotto da PlatformIO. Lo stesso binario può essere caricato sulla SuperMini. [Configurazione dei file Wokwi](https://docs.wokwi.com/vscode/project-config) · [Caricamento sul web e USB Serial/JTAG](https://docs.wokwi.com/guides/esp32)

La seriale è **USB Serial/JTAG a 115200**. Nel firmware sono attivi `ARDUINO_USB_MODE=1` e `ARDUINO_USB_CDC_ON_BOOT=1`. Nel diagramma il componente `board-esp32-c3-devkitm-1` ha `serialInterface = USB_SERIAL_JTAG` e nessun collegamento ai pin del monitor seriale. La vecchia configurazione C3-01 usa TX/RX (UART0) e il compilatore web. Per il confronto C3-03 usa il nuovo diagramma e il firmware compilato localmente. Per i vettori bastano alimentazione USB e seriale.

In laboratorio individua il dispositivo con `pio device list`. Con una sola scheda connessa puoi usare `pio run -e c3_supermini -t upload` e `pio device monitor -b 115200`. Se sono collegate più schede, indica la porta della SuperMini con `--upload-port` e `--port` rispettivamente.

## 2. Motore da implementare

La testa accetta 64 feature float32 precalcolate e produce `h1[24]`, `h2[16]`, `h3[8]`, `logits[2]` nell’ordine `no_person, person`. Per il primo confronto usa le quattro tabelle F0 in `../c3-02/results/lut/` e i vettori in `../c3-02/results/reference_vectors.json`. Il contratto r2 e il riferimento Python fissano il comportamento dell’export. [Contratto §§3–7](https://github.com/RodaRoller/esp32c3-kan-person-detection/blob/17db9a9c9e3744125ea1ff8e84cf32ea942f891b/c3-02/C3-02_model_contract.md) · [Riferimento numerico](https://github.com/KuznetsovKarazin/lut-kan/blob/004139787a0fa623386ec071383b3b4ae350334f/src/kernels/lut_backend_reference.py) · [Struttura del precedente firmware C++](https://github.com/KuznetsovKarazin/lut-kan-ids/blob/b0660d023f36c965d59511c759c34b143a3fb382/hardware/ids_esp32_lut_kan/main.cpp)

- Genera array `const` in flash per ogni layer. Le quattro tabelle e i metadati occupano **479.040 B**. I **512 B** coprono soltanto i due buffer di attivazione. Con `default.csv`, ciascuna partizione applicativa è di **1.310.720 B**. Il log del linker e `firmware.bin` danno l’occupazione effettiva.
- Per ogni arco usa i nodi del layer, gli 11 segmenti, i 16 campioni inclusivi per segmento, la tabella `uint8` e i metadati `y_min, scale` per arco e segmento. Conserva questi ultimi in FP16 e convertili a float32 nel calcolo. Interpola linearmente. Agli estremi del dominio chiuso usa i valori tabulati. Fuori dominio azzera la sola spline e conserva il ramo SiLU.
- Applica i tre coefficienti dell’arco e accumula in float32 in ordine crescente degli ingressi. Gli output di un layer passano direttamente al successivo. Il porting riguarda solo la testa. In questa fase usa le feature salvate da Python come ingresso.
- Definisci `d = logits[1] - logits[0]`. La classe è `person` solo se `d > 0`. A parità di logit restituisci `no_person`. Rifiuta gli input contenenti NaN o infinito prima dell’inferenza. Per la demo basta stampare i due logit, il margine e la classe. L’eventuale confronto storico usa `p(person) ≥ t`. Nella validation FP32 archiviata un positivo ha `p = t`. La soglia storica resta separata dalla conformità del port.

## 3. Primo risultato da consegnare

Esegui lo stesso motore C++ prima su PC e poi nel firmware Wokwi sulla fixture pubblica F0. Per ogni vettore valido scrivi una riga JSON con `id`, `status="ok"`, gli array `h1, h2, h3, logits`, il numero `d` e la stringa `decision`. Usa almeno 9 cifre significative per i float (per esempio `%.9g`). Per un input respinto scrivi solo `id` e `status="reject"`. Il verificatore ignora il testo ordinario del boot. Eseguilo su entrambi i log con Python 3, senza librerie aggiuntive.

```sh
python verify_trace.py ../c3-02/results/reference_vectors.json pc-f0.jsonl
python verify_trace.py ../c3-02/results/reference_vectors.json wokwi-f0.jsonl
```

Ogni elemento di ogni layer deve soddisfare `abs(C++ - ref) ≤ 2e-6 × max(1, abs(ref))`. Per la classe è richiesta concordanza esatta. Il primo riscontro comprende commit o PR, log di build con le versioni, 14/14 casi F0 su PC e Wokwi, errore massimo per layer e occupazione Flash/RAM indicata da PlatformIO. Registra lo spazio dei buffer separatamente da stack e heap. Pubblica su C3-03 il link al commit e ai risultati.

Poi usa in locale il checkpoint M-hist già ricevuto e i 19 vettori privati di r2 (16 validi, 3 da rifiutare). Il verificatore accetta lo stesso formato di traccia con `../private/c3-02/reference_vectors_mhist.json` come primo argomento. Verifica l’hash del file rispetto all’indice r2. Tieni in locale i pesi e i loro derivati, inclusi header, binari, tracce complete e progetti Wokwi che li incorporano. Per il relatore bastano commit, hash, riepilogo dei test e scarti. [Istruzioni per rigenerare i vettori](https://github.com/RodaRoller/esp32c3-kan-person-detection/blob/17db9a9c9e3744125ea1ff8e84cf32ea942f891b/c3-02/README.md) · [Consegna in Trello](https://trello.com/c/BnojMuIO)

Il verificatore è stato provato su una traccia Python rigenerata dalle quattro LUT pubbliche F0 (14/14), su un pareggio e su input non finiti. I log C++ su PC e Wokwi e le misure del firmware VWW costituiscono la tua consegna.
