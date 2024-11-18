import streamlit as st
import whisper
import moviepy.editor as mp
import os
from gtts import gTTS
from pydub import AudioSegment
from transformers import MarianMTModel, MarianTokenizer

# Dictionary of languages and MarianMT models (international languages)
marian_models = {
    'french': 'Helsinki-NLP/opus-mt-en-fr',
    'spanish': 'Helsinki-NLP/opus-mt-en-es',
    'german': 'Helsinki-NLP/opus-mt-en-de',
    'italian': 'Helsinki-NLP/opus-mt-en-it',
    'russian': 'Helsinki-NLP/opus-mt-en-ru',
    'chinese': 'Helsinki-NLP/opus-mt-en-zh',
    'japanese': 'Helsinki-NLP/opus-mt-en-ja',
    'portuguese': 'Helsinki-NLP/opus-mt-en-pt',
    'arabic': 'Helsinki-NLP/opus-mt-en-ar',
    'korean': 'Helsinki-NLP/opus-mt-tc-big-en-ko',
    'hindi': 'Helsinki-NLP/opus-mt-en-hi',
}

# Supported languages for gTTS
gtts_supported_languages = {
    'french': 'fr',
    'spanish': 'es',
    'italian': 'it',
    'portuguese': 'pt',
    'russian': 'ru',
    'chinese': 'zh',
    'japanese': 'ja',
    'arabic': 'ar',
    'korean': 'ko',
    'hindi': 'hi',
}

# Load the Whisper model
model = whisper.load_model("small")

# for error message
def home():
    try:
        # Code that may fail (e.g., using a library)
        import some_library
       # ... additional code ...
        return "Welcome to the homepage!"
    except ImportError:
        return "Sorry!!! The server is currently down. If the error persists wear your lucks socks and try again and of the error still persists wear your lucky gloves also and maybe try Contact (digilngo.hackerstr@gmail.com)", 500

# Function to extract audio from video
def extract_audio(video_file):
    try:
        video = mp.VideoFileClip(video_file)

        # Check if the video has an audio track
        if video.audio is None:
            st.error("Error: The video does not contain any audio.")
            return None
        audio_file = "audio.wav"
        video.audio.write_audiofile(audio_file)
        return audio_file
    except Exception as e:
        st.error(f"Error extracting audio: {e}")
        return None

# Function to transcribe audio
def transcribe_audio(audio_file):
    try:
        result = model.transcribe(audio_file, task="transcribe")
        return result['text']
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None

# Function to translate text using MarianMT
def translate_text(text, target_language):
    try:
        model_name = marian_models.get(target_language)
        if model_name:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)

            translated = model.generate(**tokenizer(text, return_tensors="pt", padding=True))
            translated_text = tokenizer.decode(translated[0], skip_special_tokens=True)
            return translated_text
        else:
            st.error(f"Model not available for language: {target_language}")
            return None
    except Exception as e:
        st.error(f"Error during translation: {e}")
        return None

# Function to convert text to speech using gTTS
def text_to_speech(text, lang_code):
    try:
        tts = gTTS(text, lang=lang_code)
        tts_audio = "translated_audio.mp3"
        tts.save(tts_audio)

        # Convert mp3 to wav format using pydub
        sound = AudioSegment.from_mp3(tts_audio)
        sound.export("translated_audio.wav", format="wav")

        return "translated_audio.wav"
    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {e}")
        return None

# Function to merge audio with video
def merge_audio_with_video(video_file, audio_file):
    try:
        video = mp.VideoFileClip(video_file)
        audio = mp.AudioFileClip(audio_file)

        # Set the new audio to the video
        final_video = video.set_audio(audio)
        output_file = "translated_video.mp4"
        final_video.write_videofile(output_file)
        return output_file
    except Exception as e:
        st.error(f"Error merging audio and video: {e}")
        return None




# Streamlit UI
def main():
    st.title("DigiLingo Translate")

    # Upload video file
    video_file = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

    if video_file is not None:
        st.subheader("Original Video Preview")
        st.video(video_file)

    # Select the language
    language = st.selectbox("Choose the target language", list(gtts_supported_languages.keys()))



    # Extract audio and translate when button is clicked
    if st.button("Transcribe and Translate"):
            # Save the uploaded video
            video_path = f"uploaded_video.mp4"
            with open(video_path, mode="wb") as f:
                f.write(video_file.read())

            # Extract audio from the video
            audio_path = extract_audio(video_path)

            if audio_path:
                # Perform transcription
                st.write("Extracting audio and converting it into text")
                transcribed_text = transcribe_audio(audio_path)

                if transcribed_text:
                    # Display the transcribed text
                    st.subheader("Original Converted Text:")
                    st.write(transcribed_text)

                    # Translate the text
                    st.write("Translating text to the selected language...")
                    translated_text = translate_text(transcribed_text, language)

                    if translated_text:
                        # Display the translated text
                        st.subheader(f"Translated Text in {language.capitalize()}:")
                        st.write(translated_text)

                        # Convert translated text to speech
                        st.write("Converting translated text to speech...")
                        translated_audio = text_to_speech(translated_text, gtts_supported_languages[language])

                        if translated_audio:
                            # Merge the new audio with the original video
                            st.write("Merging translated audio with video...")
                            translated_video = merge_audio_with_video(video_path, translated_audio)

                            if translated_video:
                                # Display the translated video
                                st.subheader("Translated Video Preview:")
                                st.video(translated_video)

                                st.download_button(
                                label = "Download Translated Video",
                                data = open(translated_video, "rb").read(),
                                file_name = "translated_video.mp4",
                                mime = "video/mp4"



                                )

                                st.write("To contact mail at digilingo.hackerstr@gmail.com")
                                st.write("Digilingo dosen't take your data")
                                st.write("To change color theme click on the three dots then click on the settings and finally click on the change active theme and choose whichever colors you like.")

                                # Clean up temporary files
                                os.remove(audio_path)
                                os.remove(translated_audio)
                                os.remove(translated_video)

if __name__ == "__main__":
    main()
