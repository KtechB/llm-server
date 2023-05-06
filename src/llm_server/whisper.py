import openai

def speech_to_text(audio, model="whisper-1", language="ja")->str:
    transcription = openai.Audio.transcribe(model=model, file=audio, language=language)
    return transcription["text"]
