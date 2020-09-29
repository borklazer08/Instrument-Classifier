import subprocess

command = r"C:\\ffmpeg-20200814-a762fd2-win64-static\\bin\\ffmpeg.exe -i C:\Users\Kartikeya\Downloads\Video\saxo.mp4 -ab 160k -ac 2 -ar 44100 -vn audio_saxo.wav"

subprocess.call(command, shell=True)
