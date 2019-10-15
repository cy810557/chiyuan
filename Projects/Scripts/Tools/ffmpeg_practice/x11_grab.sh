#!/bin/bash
#ffmpeg --enable-libxcb -video_size cif -framerate 25 -i :0.0 out.mpg
#ffmpeg -f avfoundation -list_devices true -i ""
ffmpeg -f avfoundation -i 0:0 output.mkv

# list all pixel formats
ffmpeg -pix_fmts

# list available devices
ffmpeg -f avfoundation -list_devices true -i ""

# grab web camera
ffmpeg -f avfoundation -r 30 -pix_fmt yuyv422 -i "0" output.mpg

