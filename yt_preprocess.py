from pydub import AudioSegment
import subprocess




#com2=r"ffmpeg -i C:\\Users\\Kartikeya\\Downloads\\Music\\sax2h30m.mp3 C:\\Users\\Kartikeya\\Downloads\\Music\\sax2h30m.wav"



#com5 = r"ffmpeg -i C:\\Users\\Kartikeya\\Downloads\\Music\\tru1h30.mp3 C:\Users\Kartikeya\Downloads\Music\\tru1h30.wav"
#com6=r"ffmpeg -i C:\\Users\\Kartikeya\\Downloads\\Music\\vio2h50m.mp3 C:\Users\Kartikeya\Downloads\Music\\vio2h50m.wav"


#subprocess.call(com5, shell=True)
#subprocess.call(com6, shell=True)
#subprocess.call(com2, shell=True)
dirpath ="D:\IRMAS training data\yt_"
file_path = "D:\IRMAS training data\yt_\\tru1h30.wav"
signal = AudioSegment.from_wav(file_path)
duration = int(signal.duration_seconds)
t1 = 0
t2 = 3000
d=0
# process all segments of audio file
while(t2<duration*1000):  # we need 3 second intervals
    newAudio = signal[t1:t2]
    newAudio.export(dirpath + "\\tru{}.wav".format( d), format="wav")
    t1 = t2+ 7000
    t2 = t2 + 10000
    d+=1


file_path = "D:\IRMAS training data\yt_\\cla1hr.wav"
signal = AudioSegment.from_wav(file_path)
duration = int(signal.duration_seconds)
t1 = 0
t2 = 3000
d=0

# process all segments of audio file
while(t2<duration*1000):  # we need 3 second intervals
    newAudio = signal[t1:t2]
    newAudio.export(dirpath + "\\cla{}.wav".format( d), format="wav")
    t1 = t2 + 3500
    t2 = t2 + 7500
    d+=1




file_path = "D:\IRMAS training data\yt_\\vio2h50m.wav"
signal = AudioSegment.from_wav(file_path)
duration = int(signal.duration_seconds)
t1 = 0
t2 = 3000
d=0
# process all segments of audio file
while(t2<duration*1000):  # we need 3 second intervals
    newAudio = signal[t1:t2]
    newAudio.export(dirpath + "\\vio{}.wav".format( d), format="wav")
    t1 = t2 +8500
    t2 = t2 + 11500
    d+=1





file_path = "D:\IRMAS training data\yt_\\sax2h30m.wav"
signal = AudioSegment.from_wav(file_path)
duration = int(signal.duration_seconds)
t1 = 0
t2 = 3000
d=0

# process all segments of audio file
while(t2<duration*1000):  # we need 3 second intervals
    newAudio = signal[t1:t2]
    newAudio.export(dirpath + "\\sax{}.wav".format( d), format="wav")
    t1 = t2 +8300
    t2 = t2 + 11300
    d+=1
