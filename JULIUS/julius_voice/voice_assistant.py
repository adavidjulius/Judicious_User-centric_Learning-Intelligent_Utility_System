from .stt_engine import STTEngine
from .tts_engine import TTSEngine
from julius_core.inference import get_model, ask
import tempfile
import pyaudio
import wave

class VoiceAssistant:
    def __init__(self):
        self.stt = STTEngine()
        self.tts = TTSEngine()
        self.model = get_model()

    def listen_and_respond(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        RECORD_SECONDS = 5

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                        input=True, frames_per_buffer=CHUNK)
        print("Listening...")
        frames = []
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        print("Done.")
        stream.stop_stream()
        stream.close()
        p.terminate()

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            wf = wave.open(f.name, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(p.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b''.join(frames))
            wf.close()
            audio_path = f.name

        question = self.stt.transcribe(audio_path)
        print(f"You asked: {question}")
        answer = ask(self.model, question)
        print(f"JULIUS answers: {answer}")
        self.tts.speak(answer)

if __name__ == "__main__":
    assistant = VoiceAssistant()
    assistant.listen_and_respond()
