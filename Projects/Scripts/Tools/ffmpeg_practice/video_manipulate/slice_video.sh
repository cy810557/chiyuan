ffmpeg -i in.mp3 -ss 00:01:00 -to 00:01:10 -acodec copy out.mp3
ffmpeg -i in.mp3 -ss 00:01:00 -t 10 -acodec copy out.mp3
ffmpeg -i in.mp3 -ss sseof-20 -t 10 -acodec copy out.mp3

ffmpeg -i running_runnig.mp4 -ss 00:00:10 -to 00:00:30 -c copy -copyts test1_slice_ss_after.mp4`
ffmpeg -ss 00:01:00 -i in.mp4 -to 00:01:10 -c copy -copyts out.mp4

ffmpeg -i "concat:01.mp4|02.mp4|03.mp4" -c copy out.mp4"

# Software: Avideomux
