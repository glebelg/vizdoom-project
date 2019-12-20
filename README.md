# vizdoom-project

### Installing vizdoom:
* ZDoom dependencies
  * Linux
  ```bash
  sudo apt-get install build-essential zlib1g-dev libsdl2-dev libjpeg-dev \
  nasm tar libbz2-dev libgtk2.0-dev cmake git libfluidsynth-dev libgme-dev \
  libopenal-dev timidity libwildmidi-dev unzip
  ```
  * MacOS
  ```bash
  brew install cmake boost sdl2 wget
  ```
* vizdoom
  ```bash
  sudo pip3 install vizdoom
  ```
More details [here](https://github.com/mwydmuch/ViZDoom/blob/master/doc/Building.md)

### Program launch
 ```bash
 python3 main.py
 ```
 
##### Program options
* --game-mode (defauld: "D1") - one of three game modes: D1, D2, D3
* --batch-size (defauld: 64) - batch size
* --iterations (defauld: 1000) - number of training iterations per epoch
* --epochs (defauld: 5) - number of epochs
* --test-episodes (defauld: 20) - number of test episodes

You can also use makefile, which launches all game modes with default parameters:
```bash
 make
 ```
