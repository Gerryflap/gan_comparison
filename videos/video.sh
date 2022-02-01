# Can be used to convert a sequence of images into a video
ffmpeg -pattern_type glob -i '*.png' -c:v libx264 -vf "fps=30, tpad=stop_mode=clone:stop_duration=2" -pix_fmt yuv420p -start_number 0  out.mp4
