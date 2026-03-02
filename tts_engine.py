from TTS.api import TTS

tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")

def speak(text):
    file_path = "response.wav"
    tts.tts_to_file(text=text, file_path=file_path)
    return file_path
