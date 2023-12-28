from ultralytics import YOLO
import cv2
import speech_recognition as sr
from gtts import gTTS
from pydub import AudioSegment
from pydub.playback import play
from playsound import playsound
import os

model = YOLO("best (3).pt")
previous_result = []

listener = sr.Recognizer()

def talk(text):
    myobj = gTTS(text=text, lang='en', slow=False)
    mp3_file = "tts.mp3"
    myobj.save(mp3_file)
    audio = AudioSegment.from_file("tts.mp3", format="mp3")
    play(audio)
    os.remove(mp3_file)

def take_command():
    try:
        with sr.Microphone() as source:
            print('listening...')
            listener.adjust_for_ambient_noise(source)
            voice = listener.listen(source, timeout=5)
            command = listener.recognize_google(voice)
            command = command.lower()
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Could not request results; {e}")
        return ""
    return command

def run_alexa():
    command = take_command()
    print(f"Recognized command: {command}")
    if 'detect' in command:
        results = model.predict(source=0, show_conf=False, verbose=False, stream=True, show=True)
        for r in results:
            for c in r.boxes.cls:
                class_index = int(c)
                if 0 <= class_index < len(model.names):
                    current_result = model.names[class_index]
                    if current_result not in previous_result:
                        print(current_result)
                        talk(current_result)
                        previous_result.append(current_result)
                       

while True:
    run_alexa()
