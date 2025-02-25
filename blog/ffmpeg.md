+++
date = '2025-02-06T13:38:55Z'
draft = false
title = 'FFmpeg: A Practical Guide to Essential Command-Line Options'
categories = ['Video Processing', 'FFmpeg', 'Command Line', 'Multimedia'] # Added Categories
tags = ['ffmpeg', 'video processing', 'command line', 'multimedia', 'video editing'] # Improved Tags
+++


### Introduction

[FFmpeg](https://ffmpeg.org/) is an incredibly versatile command-line tool for manipulating audio and video files. This post provides a practical collection of useful FFmpeg commands for common tasks.


### FFmpeg Command Structure

The general structure of an FFmpeg command is:

```
ffmpeg [global_options] {[input_file_options] -i input_url} ... {[output_file_options] output_url} ...
```

### Merging Video and Audio


#### Merging video and audio, with audio re-encoding

```
ffmpeg -i video.mp4 -i audio.wav -c:v copy -c:a aac output.mp4
```

#### Copying the audio without re-encoding

```
ffmpeg -i video.mp4 -i audio.wav -c copy output.mkv
```

**Why copy audio?**


1. **Preserve Quality:** Re-encoding can degrade audio quality. Copying preserves the original audio stream.
2. **Speed:** Copying is significantly faster than re-encoding.
3. **Maintain Original Parameters:**  Avoids unintended changes to bitrate, sample rate, etc.


### Replacing audio stream

```
ffmpeg -i video.mp4 -i audio.wav -c copy output.mkv
```
This is useful for dubbing or changing the soundtrack of a video.


#### Handling Different Audio/Video Durations

```
ffmpeg.exe -ss 00:00:10  -t 5 -i "video.mp4" -ss 0:00:01 -t 5 -i "music.m4a" -map 0:v:0 -map 1:a:0 -y "out.mp4"
```
The `-shortest` option ensures the output video's duration matches the shorter of the input files.  You can also use `-t` and `-ss` to specify precise start and end times for each input if needed.

```
ffmpeg -i "video.mp4" -i "music.m4a" -c:v copy -map 0:v:0 -map 1:a:0 -shortest "out.mp4"
```

---

### Video Format Conversion and Manipulation


#### Convert Video to MP4

```
ffmpeg -i input.avi output.mp4
```
Used when converting videos between formats.

---

#### Extract Audio

```
ffmpeg -i input.mp4 -q:a 0 -map a output.mp3  # Highest quality audio
ffmpeg -i input.mp4 output.mp3 # Default quality
```

#### Convert Audio

```
ffmpeg -i input.wav output.mp3
```

#### Resize Video

```
ffmpeg -i input.mp4 -vf scale=1280:720 output.mp4  # 720p resolution
```

#### Trim Video (No Re-encoding)

```
ffmpeg -i input.mp4 -ss 00:00:30 -to 00:01:00 -c copy output.mp4
```

#### Reduce Video File Size (H.265 Encoding)

```
ffmpeg -i input.mp4 -vcodec libx265 -crf 28 output.mp4  # Adjust CRF value for quality/size tradeoff
```
A lower CRF value means higher quality (and larger file size).

#### Adjust Playback Speed

* **Increase Speed (2x):**
```
ffmpeg -i input.mp4 -filter:v "setpts=0.5*PTS" output.mp4
```

* **Decrease Speed (0.5x):**
```
ffmpeg -i input.mp4 -filter:v "setpts=2.0*PTS" output.mp4
```

#### Add Subtitles (Hardcoded)

```
ffmpeg -i input.mp4 -vf "subtitles=subs.srt" output.mp4
```

#### Merge Multiple Videos

```
ffmpeg -f concat -safe 0 -i file_list.txt -c copy output.mp4
```
`file_list.txt` should contain a list of video file paths, one per line, like this:

```
file 'video1.mp4'
file 'video2.mp4'
file 'video3.mp4'
```

#### Extract a Frame as an Image

```
ffmpeg -i input.mp4 -ss 00:00:10 -vframes 1 output.png
```

#### Convert Images to Video

```
ffmpeg -framerate 30 -i image%03d.png -c:v libx264 output.mp4  # Assumes images are named image001.png, image002.png, etc.
```

#### Remove Audio

```
ffmpeg -i input.mp4 -an output.mp4
```

#### Extract Audio Segment

```
ffmpeg -i input.mp3 -ss 00:00:30 -to 00:01:00 -c copy output.mp3
```

#### Overlay Image (Watermark)

```
ffmpeg -i input.mp4 -i logo.png -filter_complex "overlay=10:10" output.mp4  # Adjust coordinates as needed
```

#### Change Frame Rate

```
ffmpeg -i input.mp4 -r 60 output.mp4
```

#### Convert to GIF

```
ffmpeg -i input.mp4 -vf "fps=10,scale=500:-1:flags=lanczos" output.gif  # Adjust fps and scale
```

#### Add Background Music

```
ffmpeg -i input.mp4 -i audio.mp3 -map 0:v -map 1:a -c:v copy -c:a aac output.mp4 # Consider adding -shortest if audio is longer
```

#### Rotate Video

```
ffmpeg -i input.mp4 -vf "transpose=1" output.mp4  # 90 degrees clockwise. Use 2 for counter-clockwise, 3 for 180 degrees.
```

#### Extract Video Without Audio

```
ffmpeg -i input.mp4 -vn -c:v copy output.mp4
```
