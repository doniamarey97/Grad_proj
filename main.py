import base64
from datetime import datetime
from fastapi import FastAPI, UploadFile, HTTPException, Form ,File
from pydantic import BaseModel
import google.generativeai as genai
import os
from pydub import AudioSegment
import tempfile
from dotenv import load_dotenv

load_dotenv()  

app = FastAPI()

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

generation_config = {
    "temperature": 0.1,
    "top_p": 0,
    "top_k": 40,
    "max_output_tokens": 1024,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash",
    generation_config=generation_config,
)

# Temporary storage for transcriptions
transcription_cache = {}

# Function to transcribe audio
async def transcribe_audio_file(file: UploadFile):

    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")

    allowed_mime_types = ["audio/wav", "audio/mpeg", "audio/ogg", "audio/webm", "audio/opus"]

    if file.content_type not in allowed_mime_types:
        raise HTTPException(status_code=400, detail="Invalid file type. Supported types: wav, mp3, ogg, webm, opus")

    try:
        audio_bytes = await file.read()
        encoded_audio = base64.b64encode(audio_bytes).decode("utf-8")

        response = model.generate_content(
            [
                "Transcribe the following audio to text:",
                {
                    "mime_type": file.content_type,
                    "data": encoded_audio,
                },
            ]
        )

        return response.text  # Extract the transcribed text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

# Function to summarize text
async def summarize_text(text: str):
    """Function to summarize text using Gemini AI."""
    try:
        response = model.generate_content(f"Summarize this: {text}")
        return response.text  # Extract summarized text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {e}")

async def enhance_text(text: str):
    """Function to improve grammar and clarity of transcribed text."""
    try:
        response = model.generate_content(f"Improve the following text by correcting grammar, enhancing clarity, and making it more natural-sounding. "
            "Do not add new content or remove important meaning. Just return the improved version without any comments, notes, or explanation:\n\n"
            f"{text}")
        return response.text  # Extract enhanced text
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing text: {e}")
        

async def detect_topics(text: str):
    response = model.generate_content(
        f"List only the main topics discussed in this text as bullet points. No explanations:\n{text}"

    )
    return response.text


# Endpoint to transcribe and store the text

@app.post("/transcribe/")
async def transcribe(file: UploadFile = File(...)):
    try:
        transcription = await transcribe_audio_file(file)

        # 1. Extract language
        language_response =  model.generate_content(
             f"Identify the language of the following text. Respond only with the language name \n\n{transcription}"
        )
        language = language_response.text.strip()

        # 2. Extract main point
        main_point_response =  model.generate_content(
             f"What is the main point of this text? do not write any intro just give the main point:\"\n\n{transcription}"
        )
        main_point = main_point_response.text.strip()

        # 3. Extract tags
        tags_response =  model.generate_content(
            f"Extract at most 5 significant keywords from the following text. Do not write any introduction. just Provide the keywords as a Python list : [tag1 , tag2 , ...] \n\n{transcription}"

        )
        # Attempt to parse tags as list
        raw_tags = tags_response.text.strip().replace("\n", "").replace("-", "")
        tags = [tag.strip() for tag in raw_tags.split(",") if tag.strip()]

        # 4. Save with metadata
        transcription_cache["latest"] = {
            "text": transcription,
            "metadata": {
                "language": language,
                "main_point": main_point,
                "tags": tags,
                "filename": file.filename,
                "upload_date": datetime.now().isoformat()
            }
        }

        return {
            
            "transcription": transcription,
            "metadata": transcription_cache["latest"]["metadata"]
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")

# Endpoint to summarize the latest transcribed text
@app.get("/summarize/")
async def summarize_latest():
    """Fetches the latest transcribed text and summarizes it."""
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found. Please transcribe an audio file first.")

    try:
        summary = await summarize_text(transcription_cache["latest"]["text"])
        return {
            "summary": summary,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error summarizing text: {e}")


# Endpoint to enhance the latest transcribed text
@app.get("/enhance/")
async def enhance_latest():
    """Enhances the grammar and clarity of the latest transcribed text."""
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found. Please transcribe an audio file first.")

    try:
        enhanced_text = await enhance_text(transcription_cache["latest"]["text"])
        return {
            "enhanced_text": enhanced_text,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error enhancing text: {e}")


@app.get("/detect_topics/")
async def detect_topics_latest():
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found.")
    
    try:
        topics = await detect_topics(transcription_cache["latest"]["text"])
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting topics: {e}")

@app.get("/extract_tasks/")
async def extract_tasks():
    """Extracts actionable tasks from the latest transcribed text."""
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found.")

    try:
        prompt = f"""
        Extract all actionable tasks from this text:
        {transcription_cache["latest"]["text"]}

        Return only a list of tasks, each on a new line, without extra explanations.
        """
        response = model.generate_content(prompt)
        tasks = response.text.strip()
        return {"tasks": tasks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error extracting tasks: {e}")


@app.post("/speaking_analysis/")
async def speaking_analysis(file: UploadFile = File(...)):
    """
    Analyze speaking performance using audio duration and transcribed text.
    """
    if "latest" not in transcription_cache:
        raise HTTPException(status_code=400, detail="No transcribed text found. Please transcribe an audio file first.")

    try:
        # Save the uploaded audio file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=file.filename) as tmp:
            audio_bytes = await file.read()
            tmp.write(audio_bytes)
            tmp_path = tmp.name

        # Use pydub to load the audio file and get duration
        audio = AudioSegment.from_file(tmp_path)
        duration_minutes = audio.duration_seconds / 60  # more accurate than len()/1000

        # Get the transcription text and word count
        text = transcription_cache["latest"]["text"]
        word_count = len(text.split())
        words_per_minute = round(word_count / max(duration_minutes, 0.01), 2)

        return {
            "audio_duration_minutes": round(duration_minutes, 2),
            "word_count": word_count,
            "words_per_minute": words_per_minute,
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during speaking analysis: {e}")
    
