import asyncio
import requests
from fastapi import FastAPI, HTTPException, Body
from vosk import Model, KaldiRecognizer
import json
from pydub import AudioSegment
import io
import os
import pydub
import time
import tempfile


app = FastAPI()

# Замените на путь к вашей модели Vosk
VOSK_MODEL_PATH = "vosk-model-small-ru-0.22"  # Замените на ваш путь

try:
    model = Model(VOSK_MODEL_PATH)
except Exception as e:
    print(f"Ошибка загрузки модели Vosk: {e}")
    exit(1)


async def transcribe_audio(audio_url_or_path):
    try:
        if audio_url_or_path.startswith("http"):
            response = requests.get(audio_url_or_path, stream=True)
            response.raise_for_status()
            if int(response.headers.get('content-length', 0)) == 0:
                raise HTTPException(status_code=400, detail="Файл пустой")
            audio_data = response.content

            try:
                audio = AudioSegment.from_file(io.BytesIO(audio_data), format="mp3")
                audio = audio.set_frame_rate(16000)
                audio = audio.export("temp.wav", format="wav")
                with open("temp.wav", "rb") as f:
                    audio_data = f.read()
            except pydub.exceptions.CouldntDecodeError as pydub_error: # Изменение здесь
                raise HTTPException(status_code=400, detail=f"Ошибка декодирования аудио: {pydub_error}") # Изменение здесь

            recognizer = KaldiRecognizer(model, 16000)
            recognizer.AcceptWaveform(audio_data)
            result = json.loads(recognizer.FinalResult())
            time.sleep(1)
            try:
                os.remove("temp.wav")
            except OSError as os_error: # Изменение здесь
                print(f"Ошибка удаления temp.wav: {os_error}")
                pass
            return result['text']
        else:
            raise HTTPException(status_code=400, detail="Поддерживаются только URL-адреса")

    except requests.exceptions.RequestException as requests_error: # Изменение здесь
        raise HTTPException(status_code=400, detail=f"Ошибка загрузки аудио: {requests_error}") # Изменение здесь
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка распознавания речи: {e}")

def get_audio_duration_from_url(audio_url):
    try:
        with requests.get(audio_url, stream=True) as response:
            response.raise_for_status()
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
                temp_file.write(response.content)
                temp_filepath = temp_file.name

            duration = get_audio_duration_ffmpeg(temp_filepath) # используем функцию из предыдущего примера
            os.remove(temp_filepath)
            return duration
    except requests.exceptions.RequestException as e:
        raise Exception(f"Ошибка загрузки аудио: {e}")
    except Exception as e:
        raise Exception(f"Ошибка при определении длительности: {e}")

def get_audio_duration_ffmpeg(filepath):
    try:
        command = f"ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 {filepath}"
        result = os.popen(command).read().strip()
        duration = float(result)
        return duration
    except Exception as e:
        raise Exception(f"Ошибка при определении длительности: {e}")


@app.post("/asr")
async def asr(audio_data: dict = Body(...)):
    try:
        audio_url_or_path = audio_data["audio_url_or_path"]
        print("connected to ", audio_url_or_path)
        text = await transcribe_audio(audio_url_or_path)

        # response = requests.get(audio_url_or_path, stream=True)
        
        duration = get_audio_duration_from_url(audio_url_or_path)

        # Пример заполнения JSON. Замените на более сложную логику, если нужно.
        # ВАЖНО! От выбранной модели VOSK сильно зависит точность содержимого контекста text, однако более мощные модели могут потребовать гораздно больших вычислительных способностей пк
        # Для извлечения данных о gender, можно воспользоваться моделью Qwen2-Audio-7B (Создать промпт с аудио и вопросом о гендере)
        dialog = [
            {"source": "receiver", "text": text, "duration": duration, "raised_voice": True, "gender": "male"},
        ]

        result_duration = {"receiver": duration}

        return {"dialog": dialog, "result_duration": result_duration}

    except KeyError:
        raise HTTPException(status_code=422, detail="Отсутствует поле 'audio_url_or_path'")
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Внутренняя ошибка сервера: {e}")