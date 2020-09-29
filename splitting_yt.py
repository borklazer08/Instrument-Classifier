import os

from pydub import AudioSegment


if __name__=="__main__":

    dataset_path = "D:\\IRMAS training data\\test"

    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):

        # ensure we're processing a genre sub-folder level
        if dirpath is not dataset_path:

            # save genre label
            print("\nSplitting: {}".format(dirpath))

            # process all audio files in instrument sub-dir
            for f in filenames:

                # load audio file
                file_path = os.path.join(dirpath, f)
                signal = AudioSegment.from_wav(file_path)
                duration = int(signal.duration_seconds)
                t1 = 0
                t2 = 3000

                # process all segments of audio file
                for d in range(duration // 3):  # we need 3 second intervals
                    newAudio = signal[t1:t2]
                    newAudio.export(dirpath +"\\{}{}.wav".format(f, d), format="wav")
                    t1 = t2
                    t2 = t2 + 3000


