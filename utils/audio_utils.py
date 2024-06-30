import os
from pydub import AudioSegment

def crop_wav_files(source_folder, target_folder, duration=10000):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    for subdir, dirs, files in os.walk(source_folder):
        for file in files:
            if file.endswith('.wav'):
                file_path = os.path.join(subdir, file)
                target_subdir = subdir.replace(source_folder, target_folder)
                if not os.path.exists(target_subdir):
                    os.makedirs(target_subdir)
                
                audio = AudioSegment.from_wav(file_path)
                segments = [audio[:duration], audio[duration:duration*2], audio[duration*2:]]
                
                for i, segment in enumerate(segments):
                    wav_file_path = os.path.join(target_subdir, f"{file[:-4]}_{i}.wav")
                    segment.export(wav_file_path, format="wav")
                    print(f"Cropped and saved: {wav_file_path}")