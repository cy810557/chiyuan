# reference:
# https://www.bugcodemaster.com/article/extract-images-frame-frame-video-file-using-ffmpeg 

# cut specified number of frames at HH:SS:MM.mmm:
ffmpeg -i video.webm -ss 00:00:07.000 -vframes 1 thumb.jpg  # 1 frame
ffmpeg -i running_runnig.mp4 -ss 00:00:13.000 -vframes 5 %03d.jpg # 5 frames

# cut 

