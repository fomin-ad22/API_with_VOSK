# API_with_VOSK
API on Fast API project with Vosk model (Creates an audio transcription and adds information about the performer, such as gender, time of performance, etc.)

• Сервис предоставляет возможность обработать аудифайл : Разбить звуки в нем на источники, определить их количество, время произношения звука и тд.

• Взаимодействие с сервисом осуществляется путем создания post-запроса на эндпоинт "http://localhost:8000/asr" с json данными в формате {"audio_url_or_path": "https://somelinkonaudiofilemp3"}. 

• Результатом создания запроса, будет следующая структура данных
{
    "dialog": [
        {
            "source": "receiver",
            "text": "sometextsinger",
            "duration": someduration,
            "raised_voice": true,
            "gender": "male"
        }
    ],
    "result_duration": {
        "receiver": someduration
    }
}
• Убедитесь в правильной установке всех необходимых компонентов :

pip install fastapi uvicorn vosk requests pydub

• Убедитесь в правильной установке самой модели и компонентов, необходимых для ее запуска : 

Скачайте с сайта https://alphacephei.com/vosk/models подходящую модель vosk для обработки аудио и разархивируйте модель в папку с проектом (папка с названием модели должна находится рядом в одной директории с файлом main.py).
Для работы моделей vosk, необходима установка FFmpeg : Убедитесь, что FFmpeg установлен и доступен в вашей системе, а если его нет, скачайте с сайта https://ffmpeg.org/ подходящую вашей системе версию и установите, после после чего добавьте в переменные среды путь к папке bin внутри FFmpeg. 
Если вы хотите использовать модель, отличную от "vosk-model-small-ru-0.22", то также нужно будет изменить содержимое переменной VOSK_MODEL_PATH в файле main.py на название модели, которую хотите использовать (папка рядом с main.py будет называться также как и модель).

• ВАЖНО ! 
От выбора модели сильно зависит точность обработки аудиофайла.
Для использования более мощных моделей Vosk, требуется гораздо больше вычислительных ресурсов вашего пк (Учитывая особенности моделей Vosk, требуется процессор с мощностью выше intel core i7 - 11 серии, а также от 16 Гб оперативной памяти), в противном случае модель не будет работать и сервис не запустится.
Если вы хотите адаптировать дополнительные возможности аналиа аудио для извлечения данных из него, необходимо использовать дополнительные LLM модели и библотеки для анализа (К примеру Qwen2-Audio-7B может подсчитать гендер в аудио, при создании промпт-запроса с аудио и требованием узнать гендер)

Обратная связь t.me - fomin_ad22
