from pathlib import Path

import numpy as np
from PIL import Image
from scipy.io.wavfile import write
from transformers import pipeline


IMAGE_MODEL = "Salesforce/blip-image-captioning-base"
VOICE_MODEL = "facebook/mms-tts-eng"

FOLDER = Path(__file__).parent
INPUT = FOLDER / "input"
OUTPUT = FOLDER / "output"
OUTPUT.mkdir(exist_ok=True)


def make_voice(text, filename, voice_model):

    audio = voice_model(text)
    sound = np.array(audio["audio"]).squeeze()
    sound = (np.clip(sound, -1, 1) * 32767).astype(np.int16)

    filename = Path(filename).stem or "description"
    path = OUTPUT / (filename + ".wav")
    write(str(path), audio["sampling_rate"], sound)
    print("Аудио сохранено:", path)


def analyze(filename, image_model, voice_model, ask_name=False):

    path = Path(filename.strip().strip('"'))
    if not path.is_file():
        path = FOLDER / path
    if not path.is_file():
        path = INPUT / Path(filename.strip().strip('"')).name

    if not path.is_file():
        print("Ошибка: фотография не найдена.")
        return None

    if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
        print("Ошибка: нужен JPG, JPEG или PNG.")
        return None

    try:
        image = Image.open(path).convert("RGB")
        print("Анализ фотографии...")
        result = image_model(image)
        description = result[0]["generated_text"].strip()

        if not description:
            print("Модель не смогла создать описание.")
            return None

        print("Фотография:", path.name)
        print("Описание:", description)

        name = path.stem
        if ask_name:
            custom_name = input("Имя аудио (Enter = как фото): ").strip()
            if custom_name:
                name = custom_name

        print("Генерация голоса...")
        make_voice(description, name, voice_model)

        with open(FOLDER / "descriptions.txt", "a", encoding="utf-8") as file:
            file.write(path.name + ": " + description + "\n")

        print("Модель картинки:", IMAGE_MODEL)
        print("Модель голоса:", VOICE_MODEL)
        print("Готово!\n")
        return description, name

    except Exception as error:
        print("Ошибка при обработке:", error)
        return None


def main():
    image_model = None
    voice_model = None
    last_text = ""
    last_name = ""

    while True:
        print("\n=== AI VISION & VOICE ===")
        print("1. Анализ одной фотографии")
        print("2. Анализ всех фотографий из input")
        print("3. Повторно озвучить описание")
        print("0. Выход")
        choice = input("Выбор: ").strip()

        if choice == "0":
            break

        if choice not in ("1", "2", "3"):
            print("Неизвестный пункт меню.")
            continue

        if choice == "3" and not last_text:
            print("Сначала проанализируй фотографию.")
            continue

        if image_model is None:
            try:
                print("Загрузка модели изображений...")
                image_model = pipeline("image-to-text", model=IMAGE_MODEL, device=-1)
                print("Загрузка голосовой модели...")
                voice_model = pipeline("text-to-speech", model=VOICE_MODEL, device=-1)
            except Exception as error:
                print("Не удалось загрузить модели:", error)
                break

        if choice == "1":
            filename = input("Путь к фото (например, cat.jpg): ")
            result = analyze(filename, image_model, voice_model, True)
            if result is not None:
                last_text, last_name = result

        elif choice == "2":
            found = False
            for path in sorted(INPUT.iterdir()):
                if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    found = True
                    result = analyze(str(path), image_model, voice_model)
                    if result is not None:
                        last_text, last_name = result
            if not found:
                print("В папке input нет фотографий.")

        elif choice == "3":
            name = input("Имя нового аудио (Enter = repeat): ").strip()
            if not name:
                name = last_name + "_repeat"
            try:
                make_voice(last_text, name, voice_model)
            except Exception as error:
                print("Ошибка генерации голоса:", error)


if __name__ == "__main__":
    main()
