+++
date = '2025-02-05T12:39:16Z'
draft = false  
title = 'Harnessing the Power of Stable Diffusion WebUI'
categories = ['AI', 'Stable Diffusion', 'Image Generation', 'Open Source'] 
tags = ['AI art', 'image generation', 'Stable Diffusion', 'WebUI'] 
+++

### Summary

In this blog I aim to try building using open source tools where possible. The benefits are price, control, knowledge and eventually quality.
In the shorter term though the quality will trail the paid versions. 
My belief is we can construct AI applications to be `self correcting` sort of like how your camera auto focuses for you. This process will involve a lot of computation so using a paid service could be costly. This for me is the key reason to choose solutions using free tools.


### stable-diffusion-webui

[stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) is a web application for stable diffusion.

I will be using it a lot in the next few posts.
The `webui` has a lot of options and extensions that  allow you to generate a lot of different images. For what I want to do I will need to install it and open up the api.


### **Getting Started: Installation**

The installation process can vary slightly depending on your operating system, but generally involves these steps:

1. **Install Python:** Ensure you have a compatible version of Python installed (typically Python 3.10 or higher).
2. **Install Git:** Git is required for cloning the WebUI repository.
3. **Clone the Repository:** Clone the Stable Diffusion WebUI repository from GitHub to your local machine using Git.
4. **Install Dependencies:** Navigate to the cloned directory and run the provided script to install the necessary Python packages.
5. **Download Stable Diffusion Model:** Download a Stable Diffusion model checkpoint file (e.g., from Civitai or Hugging Face) and place it in the designated folder within the WebUI directory.


### **Installing hf_transfer**

For me when I cloned the repo I needed to also install `hf_transfer` so that the system could download the required models to run.

1. active the environment

```bash
# where you cloned the repo
cd c:\sd.webui   
# from there activate the env
.\webui\venv\Scripts\activate
# you should see the venv activate
```

2. Install hf_transfer

```bash
# install hf_transfer
pip install hf_transfer
```

3. Enable the api

Just edit the `sd.webui\webui\webui-user.bat` file and add `--api`  to the COMMANDLINE_ARGS. IT should look like this:

```bash
@echo off

set PYTHON=
set GIT=
set VENV_DIR=
set COMMANDLINE_ARGS=--api

call webui.bat

```

### Calling the api to generate images

Here's a Python example demonstrating how to use the API:

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

