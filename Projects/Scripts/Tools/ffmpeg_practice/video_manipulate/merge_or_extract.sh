# extract video with no audio
ffmpeg -i <infile> -vcodec copy -an <outfile.mp4>

# extract audio from video
ffmpeg -i <infile> -acodec copy -vn <outfile.m4a>

# merge
ffmpeg -i <in.mp4> -i <in.m4a> -c copy out.mp4
# how to add audio on top of a video file with sound?
