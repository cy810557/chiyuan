#!/bin/bash
#ffmpeg --enable-libxcb -video_size cif -framerate 25 -i :0.0 out.mpg
#ffmpeg -f avfoundation -list_devices true -i ""
ffmpeg -f avfoundation -i 0:0 output.mkv

