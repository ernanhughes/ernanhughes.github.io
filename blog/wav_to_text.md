+++
date = '2025-01-31T11:09:13Z'
title = 'Creating AI-Powered Paper Videos: From Research to YouTube'
categories = ['Ollama', 'Whisper', 'FFmpeg', 'Stable Diffusion']
tags = ['Ollama', 'Whisper', 'FFmpeg', 'Stable Diffusion']
+++

### Summary

This post demonstrates how to automatically transform a scientific paper (or any text/audio content) into a YouTube video using AI.  We'll leverage several powerful tools, including large language models (LLMs), Whisper for transcription, Stable Diffusion for image generation, and FFmpeg for video assembly. This process can streamline content creation and make research more accessible.

### Overview

Our pipeline involves these steps:

1.  **Audio Generation (Optional):** If starting from a text document, we'll use a text-to-speech service (like NotebookLM, or others) to create an audio narration.
2.  **Transcription:** We'll use Whisper to transcribe the audio into text, including timestamps for each segment.
3.  **Database Storage:** The transcribed text, timestamps, and metadata will be stored in an SQLite database for easy management.
4.  **Text Chunking:** We'll divide the transcript into logical chunks (e.g., by sentence or time duration).
5.  **Concept Summarization:** An LLM will summarize the core concept of each chunk.
6.  **Image Prompt Generation:** Another LLM will create a detailed image prompt based on the summary.
7.  **Image Generation:** Stable Diffusion (or a similar tool) will generate images from the prompts.
8.  **Video Assembly:** FFmpeg will combine the images and audio into a final video.


### Prerequisites

*   **Hugging Face CLI:** Install it to download the Whisper model: `pip install huggingface_hub`
*   **Whisper:**  Install the `whisper-timestamped` package, or your preferred Whisper implementation.
*   **Ollama:** You'll need a running instance of Ollama to access the LLMs.
*   **Stable Diffusion WebUI (or similar):**  For image generation.
*   **FFmpeg:** For video and audio processing. Ensure it's in your system's PATH.
*   **Python Libraries:** Install necessary Python packages: `pip install pydub sqlite3 requests Pillow` (and any others as needed).


### 1. Audio Generation (Optional)

If you're starting with a text document, you'll need to convert it to audio. Several cloud services and libraries can do this.  For this example, we'll assume you have an audio file (`audio.wav`).


### 2. Transcription with Whisper

This will extract the text to a large json file

```python

import whisper_timestamped as whisper
import io

audio = whisper.load_audio("../data/Stabilizing Large Sparse Mixture-of-Experts Models.wav")

model = whisper.load_model("NbAiLab/whisper-large-v2-nob", device="cuda")

result = whisper.transcribe(model, audio, language="en")


with io.open('data.json', 'w', encoding='utf-8') as f:
  f.write(json.dumps(result, ensure_ascii=False))

```


### 3. Database Storage

Here we take the result from the process and save it to an sqlite database.

- I created two tables `transcriptions` for the text data and segments for the processed audio text.
- Optionally we could add another table for the individual words

```python
import json
import sqlite3

DB_NAME = "transcriptions.db"

def create_tables():
    """Creates SQLite tables for storing transcription data and segments separately."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    # Table for transcription metadata
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transcriptions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        text TEXT,
        language TEXT
    );
    """)

    # Table for individual segments linked to transcriptions
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS segments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        transcription_id INTEGER,
        start REAL,
        end REAL,
        text TEXT,
        tokens TEXT,
        temperature REAL,
        avg_logprob REAL,
        compression_ratio REAL,
        no_speech_prob REAL,
        confidence REAL,
        words TEXT,
        FOREIGN KEY (transcription_id) REFERENCES transcriptions(id) ON DELETE CASCADE
    );
    """)

    conn.commit()
    conn.close()
    print("Tables created successfully.")

# Run this first to create the tables
create_tables()


def insert_transcription(data):
    """Inserts transcription metadata and segments into separate tables."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    
    print("Text:", data["text"])

    # Insert into transcriptions table
    cursor.execute("""
    INSERT INTO transcriptions (text, language) 
    VALUES (?, ?)""",
    (data.get("text", ""), data.get("language", ""))
    )

    # Get the last inserted transcription ID
    transcription_id = cursor.lastrowid

    # Insert segments
    for segment in data.get("segments", []):
        cursor.execute("""
        INSERT INTO segments (
            transcription_id, start, end, text, tokens, temperature, 
            avg_logprob, compression_ratio, no_speech_prob, confidence, words
        ) 
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transcription_id,
            segment.get("start", 0),
            segment.get("end", 0),
            segment.get("text", ""),
            json.dumps(segment.get("tokens", [])),  # Store as JSON string
            segment.get("temperature", 0),
            segment.get("avg_logprob", 0),
            segment.get("compression_ratio", 0),
            segment.get("no_speech_prob", 0),
            segment.get("confidence", 0),
            json.dumps(segment.get("words", []))  # Store words as JSON string
        ))

    conn.commit()
    conn.close()
    print(f"Transcription and {len(data.get('segments', []))} segments inserted successfully.")


# Insert the result from whisper
insert_transcription(data)


# sanity check to make sure we are actually storing the data
import pandas as pd
conn = sqlite3.connect(DB_NAME)
query = "SELECT s.start, s.end, SUBSTR(s.text, 1, 50) AS text_preview FROM segments s JOIN transcriptions t ON s.transcription_id = t.id LIMIT 5"
df = pd.read_sql(query, conn)
print(df.head(5))

```

```

transcription_id  start    end                           text
0                 1   0.00  31.36   All right, so today we're go
1                 1  31.36  36.08   Each one incredibly good at 
2                 1  36.16  47.08   OK, so it's less like one gi
3                 1  47.08  53.11   Earlier attempts at this kin
4                 1  53.11  57.81   reliable and also adaptable,
```

### 4. Handling Large Audio Files with Pydub

If your audio file is large you may want to split it into chunks for processing. You can use [pydub](https://github.com/jiaaro/pydub) to achieve this 

```python
from pydub import AudioSegment
from pydub.silence import split_on_silence

# Load audio file
audio = AudioSegment.from_wav("large_audio.wav")

# Split audio on silence
chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)

# Process each chunk
for i, chunk in enumerate(chunks):
    chunk.export(f"chunk_{i}.wav", format="wav")
```

You can then transcribe each chunk separately using Whisper.

### 5. Text Chunking

If the text is really big we can chink it.

```python
import sqlite3

# ... (get_text_chunks function as in your original post) ...

transcription_id = 1  # Replace with actual ID
chunks = get_text_chunks(transcription_id, max_chunk_duration=7.0)

for chunk in chunks:
    print(f"[{chunk['start']} - {chunk['end']} sec]: {chunk['text']}")
```


### 7. Slowing down pronunciation for better text generation

In some cases you may need to slow down some text to help the LLMs process it better. You can use ffmpeg to do this

```bash
ffmpeg -i input.mp3 -filter:a "atempo=0.9" output.mp3
```


### Splitting the transcript into chunks of text

- The segments table contains individual text fragments, each with a start and end timestamp.
- We will group these segments into reasonable chunks based on a configurable duration (e.g., 5–10 seconds).
- If a chunk ends in the middle of a sentence, we'll extend it to include the next segment.


```python 

import sqlite3

def get_text_chunks(transcription_id: int, max_chunk_duration: float = 5.0):
    """
    Retrieve reasonable chunks of text from the database, grouping segments into logical sentence structures.

    Args:
        transcription_id (int): The transcription ID to query.
        max_chunk_duration (float): Maximum duration (in seconds) for each chunk.

    Returns:
        list of dict: A list of text chunks with metadata.
    """
    conn = sqlite3.connect("transcriptions.db")
    cursor = conn.cursor()

    # Fetch segments sorted by start time
    cursor.execute("""
        SELECT id, start, end, text
        FROM segments
        WHERE transcription_id = ?
        ORDER BY start ASC
    """, (transcription_id,))

    segments = cursor.fetchall()
    conn.close()

    chunks = []
    current_chunk = []
    current_start = None
    current_end = None
    current_duration = 0

    for seg in segments:
        seg_id, seg_start, seg_end, seg_text = seg
        seg_duration = seg_end - seg_start

        # If chunk is empty, initialize it
        if not current_chunk:
            current_start = seg_start
            current_end = seg_end
            current_duration = seg_duration
            current_chunk.append(seg_text)
            continue

        # Check if adding this segment exceeds the max chunk duration
        if current_duration + seg_duration > max_chunk_duration:
            # Finalize the current chunk before starting a new one
            chunks.append({
                "start": current_start,
                "end": current_end,
                "text": " ".join(current_chunk)
            })

            # Start a new chunk
            current_chunk = [seg_text]
            current_start = seg_start
            current_end = seg_end
            current_duration = seg_duration
        else:
            # Extend the current chunk
            current_chunk.append(seg_text)
            current_end = seg_end
            current_duration += seg_duration

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append({
            "start": current_start,
            "end": current_end,
            "text": " ".join(current_chunk)
        })

    return chunks

# Example usage
transcription_id = 1  # Replace with actual transcription ID
chunks = get_text_chunks(transcription_id, max_chunk_duration=7.0)

# Print results
for chunk in chunks:
    print(f"[{chunk['start']} - {chunk['end']} sec]: {chunk['text']}")

```

```
[0.0 - 31.36 sec]:  All right, so today we're going to be looking at AI and specifically how to make it a whole lot smarter, but without needing, you know, like a giant supercomputer. You're interested in these sparse expert models, right? And specifically this paper about STMOe, stable and transferable mixture of experts. It sounds kind of intimidating. I think the idea is actually really elegant. It is. Think about it this way. Instead of one massive AI brain, you know, trying to process everything. What if you had a team of specialized experts? What if you had a team of people who were able to make a smart AI? What if you had a team of people who were able to make a smart computer? What if you had a team of people who were able to make a smart computer? What if you had a team of people who were able to make a smart computer? What if you had a
[31.36 - 36.08 sec]:  Each one incredibly good at their own thing. That's the core concept behind these sparse expert models.
[36.16 - 47.08 sec]:  OK, so it's less like one giant dictionary. More like having a linguist, a grammarian, a poet all working together. Yeah, that's a great analogy. And the ST part is really key here. Stable and transferable.
[47.08 - 53.11 sec]:  Earlier attempts at this kind of AI were, well, a bit temperamental. STMOe is designed to be more
[53.11 - 57.81 sec]:  reliable and also adaptable, meaning you can train it on one task and then easily apply it
```


### 8. Concept Summarization and Image Prompt Generation

We have configurable chunks of text. Now we want to get a concept for each of these chunks. We will use a Large Language model to do this.

First we need to structure a prompt to get the summary

```
Summarize the following text into a single, overarching concept.  Focus on the main idea, even if multiple topics are touched upon.  Provide a concise, one or two-sentence summary of this core concept.

Text:
{text_chunk}

```

We will also create a method to call ollama to get a response to a prompt

```python

import logging
import requests

def chat_with_ollama(prompt, model_name="qwen2.5", ollama_base_url="http://localhost:11434"):
    """Chat with Ollama."""
    try:
        url = f"{ollama_base_url}/api/generate"
        data = {
            "prompt": prompt,
            "model": model_name,
            "stream": False
        }
        response = requests.post(url, json=data)
        
        # Check if the request was successful
        if response.status_code == 200:
            # Parse the JSON response
            response_json = response.json()
            print("Chat Response:")
            pretty_json = json.dumps(response_json, indent=4)
            logging.info(pretty_json)
            result = response_json["response"]
            print(f"For prompt: {prompt}\n result: {result}")
            return response_json["response"]
        else:
            print(f"Failed to get responnse from ollama. Status code: {response.status_code}")
            print("Response:", response.text)
            return None
    
    except requests.ConnectionError:
        print("Failed to connect to the Ollama server. Make sure it is running locally and the URL is correct.")
        return None
    except json.JSONDecodeError:
        print("Failed to parse JSON response from Ollama server.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

```


### 9. Convert the summary concept into an prompt for an image

So we have a summary now we want to create a background image for this summary. We use a LLM here again. First we need a good prompt for this task.

```
Create a detailed and imaginative image prompt for the following text summary.  The prompt should be suitable for a high-quality image generation model like Stable Diffusion or DALL-E 3. Be specific about the style, composition, and key elements of the image. Aim for a visually compelling and evocative description that captures the essence of the summary.

Summary:
{summary}

Image Prompt:

```

#### Store these new results into a database table

To organize this process and to allow for better processing we will store all these results into a database table.


```python
import requests
import json


def store_summary_in_db(text, summary, image_prompt, db_name="summaries.db"):
    """
    Stores the original text and its summary in an SQLite database.

    Args:
        text: The original text.
        summary: The summarized concept.
        db_name: The name of the SQLite database file.
    """

    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS text_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                original_text TEXT NOT NULL,
                summary TEXT NOT NULL,
                image_prompt TEXT  -- New column for image prompt
           )
        ''')

        # Insert the text and summary
        cursor.execute("INSERT INTO text_summaries (original_text, summary, image_prompt) VALUES (?,?,?)", (text, summary, image_prompt))

        conn.commit()
        print(f"Stored summary in database: {db_name}")

    except sqlite3.Error as e:
        print(f"Error accessing SQLite database: {e}")
    finally:
        if conn:
            conn.close()


def summarize_text_concept(text_chunk):  
    """
    Summarizes the core concept of a text chunk using Ollama.

    Args:
        text_chunk: The text to summarize.
        ollama_base_url: The base URL of your Ollama instance.

    Returns:
        A string summarizing the main concept, or None if there's an error.
    """

    try:
        # Construct the prompt for Ollama.  This prompt is crucial for getting a concise, single-concept summary.
        prompt = f"""
        Summarize the following text into a single, overarching concept.  Focus on the main idea, even if multiple topics are touched upon.  Provide a concise, one or two-sentence summary of this core concept.

        Text:
        {text_chunk}

        Summary:
        """

        summary = chat_with_ollama(prompt)


        image_prompt_request = f"""
        Create a detailed and imaginative image prompt for the following text summary.  The prompt should be suitable for a high-quality image generation model like Stable Diffusion or DALL-E 3. Be specific about the style, composition, and key elements of the image. Aim for a visually compelling and evocative description that captures the essence of the summary.

        Summary:
        {summary}

        Image Prompt:
        """

        image_prompt = chat_with_ollama(image_prompt_request)


        store_summary_in_db(prompt, summary, image_prompt)

        return summary, image_prompt

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Ollama: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response from Ollama: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


# Print results
for chunk in chunks:
   summarize_text_concept(chunk['text'])
```

#### Example results


| Initial prompt | Summary | Image Propmpt|
| ----- | ----- | ----- |
| Summarize the following text into a single, overarching concept.  Focus on the main idea, even if multiple topics are touched upon.  Provide a concise, one or two-sentence summary of this core concept.      Text: It is. They tracked individual words as they were processed by the different experts, creating a visual map of the AI's thought process, so to speak. And what they discovered was fascinating.  Summary:|The main idea is that an analysis of how AI processes language revealed unexpected insights through visual maps of its thought processes. | Image Prompt: Title: "Neural Networks Unveiled: AI's Thought Process Through Visual Maps" Style: A high-definition, vibrant digital   llustration with a sleek and modern aesthetic. The style should combine elements of scientific data visualization with the artistic flair of concept art to create an engaging and immersive scene. Composition: - Center Stage: At the center of the image is a large, intricate neural network diagram, representing AI's thought processes. This neural network is depicted as a 3D construct, with interconnected nodes glowing in various hues of blue and green, symbolizing different levels of activity. - Surrounding Elements: Around the central neural network, there are several interactive visual maps that expand and illustrate key insights. These include: - A dynamic heat map showing the areas where the AI processes language most intensely (colorful gradients from red to yellow). - An animated flowchart depicting the path an input text takes through various layers of the neural network. - A timeline or sequence chart illustrating stages of analysis and decision-making, with each step highlighted in a different color to show progression. - Background: The background is a gradient transitioning from deep space blues at the top to earthy greens and browns at the bottom, symbolizing the vast expanse of data and human understanding. In the lower third, there are subtle patterns resembling neurons or text fragments, hinting at the complexity beneath the surface.   Key Elements:  - Interactive Nodes: Some nodes in the neural network glow more brightly than others, indicating high levels of activity corresponding to specific insights discovered during analysis. - Arrow Indicators: Directional arrows link various parts of the diagram and maps, showing the flow of information and thought processes.  - Highlighted Insights: Specific areas or phrases within the visual maps are highlighted with brighter colors and bold fonts, drawing attention to key findings about AI's language processing capabilities.  Text Overlay: In the upper right corner, there is a concise text overlay stating "Analysis of AI Thought Processes" in clean, modern typography. Below this, there could be another line such as "Unveiling Unexpected Insights Through Visual Maps," further emphasizing the main theme of the image. Overall, the image should evoke a sense of discovery and curiosity about the inner workings of AI technology while also highlighting its potential for understanding complex data through innovative visualization techniques. |


### 10. Generating an image for the prompt

Next we need to generate an image for this prompt. 

You can use AUTOMATIC1111 to do this


```python
import requests
import json
import base64

def generate_image_sd_webui(prompt, sd_webui_url="http://127.0.0.1:7860"):  # Default SD WebUI URL
    """
    Generates an image using Stable Diffusion WebUI's API.

    Args:
        prompt: The image generation prompt.
        sd_webui_url: The URL of your running SD WebUI instance.

    Returns:
        The generated image as a base64 encoded string, or None on error.
    """
    try:
        payload = {
            "prompt": prompt,
            "steps": 20,  # Number of diffusion steps
            "width": 512,  # Image width
            "height": 512,  # Image height
            # ... other parameters as needed (see API docs) ...
        }

        response = requests.post(f"{sd_webui_url}/sdapi/v1/txt2img", json=payload)  # Use txt2img endpoint
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)

        r = response.json()

        if "images" in r and len(r["images"]) > 0:
            image_base64 = r["images"][0] # Get the first image (if multiple were generated)
            return image_base64
        else:
          print(f"Unexpected response format: {r}") # Handle unexpected JSON
          return None

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with SD WebUI: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"Invalid JSON response from SD WebUI: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None


def save_image(base64_image, filename="generated_image.png"):
    """Decodes and saves a base64 encoded image."""
    try:
        image_bytes = base64.b64decode(base64_image)
        with open(filename, "wb") as f:
            f.write(image_bytes)
        print(f"Image saved to {filename}")
    except Exception as e:
        print(f"Error saving image: {e}")

# Example usage:
prompt = "A majestic dragon flying over a fantasy landscape"
image_base64 = generate_image_sd_webui(prompt)

if image_base64:
    save_image(image_base64)
else:
    print("Image generation failed.")


```

#### Add metadata to image

This save will add the prompt metadata to the saved image. 

```python

def save_image_with_metadata(base64_image, prompt, filename="generated_image.png"):
    """Decodes, adds metadata, and saves a base64 encoded image."""
    try:
        image_bytes = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_bytes))  # Use BytesIO to open from memory

        # Create or get the EXIF data (if it exists)
        exif = image.getexif() or {}

        # Add the prompt as a user comment (or another EXIF tag)
        exif[37510] = prompt  # 37510 is the tag for user comment

        image.save(filename, exif=exif) # Save with EXIF data
        print(f"Image saved to {filename} with metadata.")

    except Exception as e:
        print(f"Error saving image with metadata: {e}")

```

### 11. Generating images from the database

We have saved a list of prompts now we want to query them and generate images for each of the prompts.


```python

import sqlite3
import requests
import json
import base64
from PIL import Image
from PIL.ExifTags import TAGS
import io
import os

# ... (generate_image_sd_webui and save_image_with_metadata functions remain the same) ...

def generate_images_from_db(db_name="summaries.db", sd_webui_url="http://127.0.0.1:7860"):
    """
    Retrieves prompts from the database and generates images using SD WebUI.

    Args:
        db_name: The name of the SQLite database file.
        sd_webui_url: The URL of your running SD WebUI instance.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT id, original_text, summary, image_prompt FROM text_summaries")
        rows = cursor.fetchall()

        for row in rows:
            id, original_text, summary, image_prompt = row
            print(f"Generating image for ID: {id}")

            if image_prompt:
                image_base64 = generate_image_sd_webui(image_prompt, sd_webui_url)
                if image_base64:
                    filename = f"generated_image_{id}.png"  # Include ID in filename
                    save_image_with_metadata(image_base64, image_prompt, filename)
                else:
                    print(f"Image generation failed for ID: {id}")
            else:
                print(f"No image prompt found for ID: {id}")

    except sqlite3.Error as e:
        print(f"Error accessing SQLite database: {e}")
    finally:
        if conn:
            conn.close()

```

#### Saving images to a database table

So instead of having a lot of files I store the images in an sqlite database table.


```python
import sqlite3
import requests
import json
import base64
from PIL import Image
from PIL.ExifTags import TAGS
import io

def store_image_in_db(image_data, text_id, image_prompt, db_name="images.db"):
    """
    Stores the image data along with its associated text ID and prompt in an SQLite database.

    Args:
        image_data: The binary image data.
        text_id: The ID of the corresponding text entry in the text summaries database.
        image_prompt: The prompt used to generate the image.
        db_name: The name of the SQLite database file.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        # Create the table if it doesn't exist
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text_id INTEGER NOT NULL,
                image_prompt TEXT NOT NULL,
                image_data BLOB NOT NULL,
                FOREIGN KEY (text_id) REFERENCES text_summaries(id)  -- Optional foreign key
            )
        ''')

        cursor.execute("INSERT INTO images (text_id, image_prompt, image_data) VALUES (?,?,?)",
                       (text_id, image_prompt, image_data))

        conn.commit()
        print(f"Stored image in database: {db_name}")

    except sqlite3.Error as e:
        print(f"Error accessing SQLite database: {e}")
    finally:
        if conn:
            conn.close()


def generate_images_from_db(db_name="summaries.db", sd_webui_url="http://127.0.0.1:7860"):
    """
    Retrieves prompts from the database, generates images, and stores them in a new database.

    Args:
        db_name: The name of the SQLite database file with text summaries.
        sd_webui_url: The URL of your running SD WebUI instance.
    """
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()

        cursor.execute("SELECT id, original_text, summary, image_prompt FROM text_summaries")
        rows = cursor.fetchall()

        for row in rows:
            id, original_text, summary, image_prompt = row
            print(f"Generating image for ID: {id}")

            if image_prompt:
                image_base64 = generate_image_sd_webui(image_prompt, sd_webui_url)
                if image_base64:
                    image_data = base64.b64decode(image_base64)  # Decode the image data
                    store_image_in_db(image_data, id, image_prompt)  # Store in image database
                else:
                    print(f"Image generation failed for ID: {id}")
            else:
                print(f"No image prompt found for ID: {id}")

    except sqlite3.Error as e:
        print(f"Error accessing SQLite database: {e}")
    finally:
        if conn:
            conn.close()



```

### 12. Video Assembly with FFmpeg

Here I use the database table to generate a movie using the timestamps and the images we generated.

I then merge that generated movie with the sound file we created initially.

For this process I use ffmpeg. I tried moviepy but had issues on windows.

```python

import sqlite3
import os
import io
from PIL import Image
import subprocess
import datetime

def extract_and_merge_images(images_db="images.db", transcriptions_db="transcriptions.db", output_movie="merged_movie.mp4", output_folder="img"):
    """
    Extracts images, merges them into a video using FFmpeg, respecting timestamps from transcriptions table.
    """
    try:
        conn_images = sqlite3.connect(images_db)
        cursor_images = conn_images.cursor()

        conn_transcriptions = sqlite3.connect(transcriptions_db)
        cursor_transcriptions = conn_transcriptions.cursor()

        os.makedirs(output_folder, exist_ok=True)

        # Fetch image data and timestamps from transcriptions table, ordered by ID
        cursor_transcriptions.execute("SELECT id, start, end FROM segments ORDER BY id")
        transcription_data = cursor_transcriptions.fetchall()

        image_files = []
        for transcription_id, start_time_str, end_time_str in transcription_data:
            cursor_images.execute("SELECT image_data FROM images WHERE text_id = ?", (transcription_id,))
            image_row = cursor_images.fetchone()

            if image_row:
                image_data = image_row[0]
                try:
                    image = Image.open(io.BytesIO(image_data))
                    format = image.format.lower() if image.format else "png"
                    filename = f"image_{transcription_id}.{format}"
                    filepath = os.path.join(output_folder, filename)
                    image.save(filepath)
                    print(f"Image {transcription_id} saved to {filepath}")

                    # Convert time strings to datetime.time objects (or timedelta if appropriate)
                    start_time = int(start_time_str) if start_time_str else 0
                    end_time = int(end_time_str) if end_time_str else 0

                    image_files.append((filepath, start_time, end_time))

                except Exception as e:
                    print(f"Error processing image {transcription_id}: {e}")
            else:
                print(f"No image found for transcription ID: {transcription_id}")

        # Create a text file with image filepaths and durations for FFmpeg
        with open("image_list.txt", "w") as f:
            for filepath, start_time, end_time in image_files:
                if start_time and end_time:
                    # Calculate duration from start and end times
                    duration = (end_time - start_time)
                    f.write(f"file '{filepath}'\n")
                    f.write(f"duration {duration}\n")
                else:
                    print(f"Missing start or end time for image: {filepath}")
                    duration = 5  # Default display duration
                    f.write(f"file '{filepath}'\n")
                    f.write(f"duration {duration}\n")

        # Use FFmpeg to create the video from the image list
        command = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",  # Allow unsafe file paths (if needed)
            "-i", "image_list.txt",
            "-vf", "scale=1280:720",  # Example scaling. Adjust as needed
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",  # Important for compatibility
            "temp_video.mp4"
        ]

        subprocess.run(command, check=True)  # Run FFmpeg, check for errors

        os.remove("image_list.txt")  # Clean up the list file

        audio_file="audio.wav"
        # Add audio using MoviePy
        # Use FFmpeg to add the audio to the video
        audio_command = [
            "ffmpeg",
            "-i", "temp_video.mp4",
            "-i", audio_file,
            "-c:v", "copy",  # Copy video stream (no re-encoding)
            "-c:a", "aac",  # Encode audio to AAC
            "-shortest", # Use shortest stream as reference
            output_movie
        ]
        subprocess.run(audio_command, check=True)

        os.remove("temp_video.mp4")  # Clean up temporary file

        print(f"Video with audio created: {output_movie}")

    except (sqlite3.Error, OSError, ValueError, subprocess.CalledProcessError) as e:
        print(f"An error occurred: {e}")
    finally:
        if conn_images:
            conn_images.close()
        if conn_transcriptions:
            conn_transcriptions.close()


# Example usage:
extract_and_merge_images()
```


### End Result

{{<youtube ZCuJUv3WD1M>}}



### **Code Examples**

Check out the [wave_to_text](https://github.com/ernanhughes/wave_to_text) repo for the code used in this post and additional examples.
