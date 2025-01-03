# Text to Video

**How to Use the Repository**

Text-To-Video-AI is designed to be simple to use while offering a range of customization options. Below is a detailed guide on how to utilize the repository, including a Python code example.

---

**1. Clone the Repository**

First, clone the repository to your local machine:

```bash
git clone https://github.com/SamurAIGPT/Text-To-Video-AI.git
cd Text-To-Video-AI
```

---

**2. Install Dependencies**

The repository includes a `requirements.txt` file for installing all the necessary dependencies:

```bash
pip install -r requirements.txt
```

---

**3. Configure API Keys**

The tool requires API keys from OpenAI and Pexels for generating content. You can obtain these keys by signing up for their respective platforms.

Set the keys as environment variables:

```bash
export OPENAI_KEY="your_openai_api_key"
export PEXELS_KEY="your_pexels_api_key"
```

Alternatively, you can create a `.env` file in the project directory and add the keys:

```
OPENAI_KEY=your_openai_api_key
PEXELS_KEY=your_pexels_api_key
```

---

**4. Run the Application**

To generate a video, run the application with a topic or text prompt:

```bash
python app.py "Your Topic Here"
```

The application will process the input, retrieve relevant images or video clips, and synthesize a video complete with narration.

![Text to video flow](img\text_to_video_process.png "Text to video flow")
---

**Features of Text-To-Video-AI**

1. **Natural Language Understanding**: Converts text into meaningful visual and audio content.
2. **Voice Synthesis**: Generates lifelike narration for the video.
3. **Customizable Themes**: Users can specify styles, tones, and durations for the video.
4. **Image and Clip Integration**: Automatically fetches relevant visuals from platforms like Pexels.
5. **Flexible Output**: Generates videos in high-quality MP4 format.

---

**Python Code Example**

Here’s an example script that demonstrates how to use the core functionality of Text-To-Video-AI:

```python
import os
from text_to_video import TextToVideo

# Set your API keys
os.environ["OPENAI_KEY"] = "your_openai_api_key"
os.environ["PEXELS_KEY"] = "your_pexels_api_key"

# Initialize the TextToVideo class
video_generator = TextToVideo(
    openai_key=os.getenv("OPENAI_KEY"),
    pexels_key=os.getenv("PEXELS_KEY")
)

# Define the topic or script
topic = "The impact of climate change on our planet"

# Generate the video
output_file = video_generator.generate_video(topic)

# Print the output file location
print(f"Video generated and saved as: {output_file}")
```

---

**Quick Start in Google Colab**

If you prefer not to set up the project locally, you can use the provided Google Colab notebook. Follow these steps:

```Python
# Google Clolab example

# Install Python

!sudo apt-get update -y
!sudo apt-get install -y python3.11
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
!sudo update-alternatives --config python3
!sudo apt-get install -y python3.11-distutils
!wget https://bootstrap.pypa.io/get-pip.py
!python3.11 get-pip.py

# clone the Text-Video-AI repo

!git clone https://github.com/SamurAIGPT/Text-To-Video-AI

# change directory into the repo

%cd Text-To-Video-AI

# install dependencies
!pip3.11 install -r requirements.txt

# here we set the API keys use teh key icon at the side in google colab to add key values

import os
from google.colab import userdata
os.environ["OPENAI_KEY"] = userdata.get('OPENAI_KEY')
os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
os.environ["PEXELS_KEY"] = userdata.get('GROQ_API_KEY')

# We also need ImageMagick
!apt install imagemagick &> /dev/null
!sed -i '/<policy domain="path" rights="none" pattern="@\*"/d' /etc/ImageMagick-6/policy.xml


# Now we generate the topic
!python3.11 app.py "Meditation"


# Next we move and display the file
from IPython.display import HTML
from base64 import b64encode
import datetime
import shutil


def add_timestamp_to_filename(filename):
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Separate the filename and its extension
    name, ext = filename.rsplit(".", 1)
    
    # Add the timestamp and reassemble the filename
    new_filename = f"{name}_{timestamp}.{ext}"
    return new_filename

# Path to the video file
original_filename = "rendered_video.mp4"
video_path = add_timestamp_to_filename(original_filename)
print(f"video path: {video_path}")

# Move the file
try:
    shutil.move(original_filename, video_path)
    print(f"File moved from {original_filename} to {video_path}")
except FileNotFoundError:
    print(f"The source file {original_filename} does not exist.")
except Exception as e:
    print(f"An error occurred: {e}")



# Function to display video
def display_video(video_path, width=640, height=480):
    # Load video
    video_file = open(video_path, "rb").read()
    video_url = "data:video/mp4;base64," + b64encode(video_file).decode()
    return HTML(f"""
    <video width={width} height={height} controls>
        <source src="{video_url}" type="video/mp4">
    </video>
    """)

# Display video
display_video(video_path)

```

[Colab Example](https://github.com/ernanhughes/youtube-videos/blob/main/Text_to_Video.ipynb)


## Generated Video 


### Meditation


[![Meditation](img\meditation.png)](https://youtu.be/A1cM6r0iPU8)


