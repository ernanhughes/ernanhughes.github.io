# AI Auto Generated Video

## Introduction
This is based upon this github: [AI-Auto-Video-Generator](https://github.com/BB31420/AI-Auto-Video-Generator)

You give the application a prompt.  
[It then uses OpenAI to improve the prompt](#generate-a-story-from-the-prompt)  
It then generates a story using OpenAI's gpt4.
Uses this story to [generate image prompts for the story](#generate-image-prompts-from-the-story)  
[Uses the prompts to generate images](#generate-images-form-the-image-prompts)  
[Add a voiceover using ElevenLabs API](#generate-voiceover)  
[Then create the video](#create-the-video)



## Generate a story from the prompt

Here we use ChatGPT to create the best possible prompt

```Python

def generate_story(prompt):
    """Generate a story based on the given prompt using OpenAI's GPT-4 model."""

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use 'gpt-4' for the best creative writing model
        messages=[
            {"role": "system", "content": "You are a creative writing assistant."},
            {
                "role": "user",
                "content": f"Can you please improve this prompt to make it produce the best story possible : {prompt}",
            },
        ],
        max_tokens=400,  # Adjust as per your desired length
        temperature=0.7,  # Controls randomness; adjust for creativity
    )

    try:
        final_story_prompt = response.choices[0].message["content"].strip()
    except IndexError:
        raise ValueError("No choices found in the response from OpenAI.")

    response = openai.ChatCompletion.create(
        model="gpt-4",  # Use 'gpt-4' for the best creative writing model
        messages=[
            {"role": "system", "content": "You are a creative writing assistant."},
            {"role": "user", "content": final_story_prompt},
        ],
        max_tokens=400,  # Adjust as per your desired length
        temperature=0.7,  # Controls randomness; adjust for creativity
    )
    story = response.choices[0].message["content"].strip()
    return story, final_story_prompt


def save_story_with_image_prompts(directory, story, prompt, image_prompts):
    with open(f"{directory}/story_{timestamp}.txt", "w") as f:
        f.write(prompt + "\n" + story + "\n\nImage Prompts:\n")
        for idx, image_prompt in enumerate(image_prompts, start=1):
            f.write(f"{idx}: {image_prompt}\n")


def save_story(directory, story):
    file_path = f"{directory}/story_{timestamp}.txt"
    print(f"Saving story to {file_path}")
    with open(file_path, "w") as f:
        f.write(story)
    return file_path  # Return the file path where the story is saved

```


## Generate image prompts from the story

```Python
def extract_image_prompts(story, num_prompts=5):
    nlp = spacy.load("en_core_web_sm")

    # Custom list of uninformative words
    uninformative_words = ["can", "to", "which", "you", "your", "that", "their", "they"]

    # Split the story into individual sentences
    doc = nlp(story)
    sentences = [sent.text.strip() for sent in doc.sents]

    # Find the main subject or noun phrase in each sentence
    main_subjects = []
    for sentence in sentences:
        doc = nlp(sentence.lower())
        for chunk in doc.noun_chunks:
            if chunk.root.dep_ == "nsubj" and chunk.root.head.text.lower() != "that":
                main_subjects.append(chunk)

    if main_subjects:
        main_subject = main_subjects[0]
    else:
        main_subject = None

    # Find the related words (adjectives, verbs) to the main subject
    related_words = defaultdict(list)
    for sentence in sentences:
        doc = nlp(sentence.lower())
        for tok in doc:
            # Avoid uninformative words and punctuation
            if tok.text in uninformative_words or not tok.text.isalnum():
                continue
            # If the token is a noun and it's not the main subject
            if (tok.pos_ == "NOUN") and (
                main_subject is None or (tok.text != main_subject.text)
            ):
                related_words[sentence].append(tok.text)

    # Create image prompts
    image_prompts = []
    for sentence, related in related_words.items():
        if main_subject is not None:
            prompt = f"{main_subject.text} {' '.join(related)} photorealistic"
        else:
            prompt = f"{sentence} photorealistic"
        image_prompts.append(prompt)

    # If we couldn't generate enough prompts, duplicate the existing ones
    if len(image_prompts) < num_prompts:
        print(
            f"Could only generate {len(image_prompts)} unique prompts out of the requested {num_prompts}. Duplicating prompts..."
        )
        i = 0
        while len(image_prompts) < num_prompts:
            image_prompts.append(image_prompts[i])
            i = (i + 1) % len(image_prompts)  # cycle through existing prompts

    print("\nGenerated Image Prompts:")
    for idx, prompt in enumerate(image_prompts, start=1):
        print(f"{idx}: {prompt}")

    # Ask the user whether they want to proceed or enter their own prompts
    user_input = input("\nDo you want to proceed with these prompts? (y/n): ")
    if user_input.lower() == "y":
        return image_prompts
    elif user_input.lower() == "n":
        user_prompts = []
        print("\nEnter your own image prompts:")
        for i in range(num_prompts):
            user_prompt = input(f"Prompt {i+1}: ")
            user_prompts.append(user_prompt)
        return user_prompts
    else:
        print(
            "Invalid input. Please enter 'y' to proceed with the generated prompts or 'n' to enter your own prompts."
        )

```


## Generate images form the image prompts

Uses openai to generate the images from the constructed prompts

```Python
def generate_images(image_prompts):
    images = []

    for prompt in image_prompts:
        response = openai.Image.create(
            prompt=prompt,
            n=1,
            size="1024x1024",
        )

        if response.data:
            image_url = response.data[0].url
            images.append(image_url)
        else:
            print(f"Error generating image for prompt '{prompt}'")
            return []
        time.sleep(12)
    return images


def save_images(directory, images, timestamp):

    for idx, image_url in enumerate(images):
        download_image(image_url, f"{directory}/image_{timestamp}_{idx}.png")


def download_image(url, filename):
    response = requests.get(url)
    with open(filename, "wb") as f:
        f.write(response.content)
```

## Generate voiceover

We use elevenlabs to generate a voice over

```Python

def generate_voiceover(story, save_file=False):
    headers = {
        "xi-api-key": os.getenv("ELEVENLABS_API_KEY"),
        "Content-Type": "application/json",
        "accept": "audio/mpeg",
    }
    data = {
        "text": story + "..Comment with your favorite fact...",
        "voice_settings": {"stability": 0.3, "similarity_boost": 0.3},
    }
    response = requests.post(
        "https://api.elevenlabs.io/v1/text-to-speech/AZnzlk1XvdvUeBnXmlld",
        headers=headers,
        json=data,
    )
    if response.status_code == 200:
        if save_file:
            with open("file.mp3", "wb") as f:
                f.write(response.content)
        return response.content
    else:
        print(
            f"Error while generating voiceover with status code {response.status_code}"
        )
        return None


def save_voiceover(directory, voiceover_content, timestamp):
    voiceover_filename = f"{directory}/voiceover_{timestamp}.mp3"
    with open(voiceover_filename, "wb") as f:
        f.write(voiceover_content)

```


## Create the video

We use each of the generated images and the voice over to build a movie.

```Python

def create_video(directory, images, voiceover_content, story, timestamp):
    # Save voiceover
    voiceover_filename = f"{directory}/voiceover_{timestamp}.mp3"
    with open(voiceover_filename, "wb") as f:
        f.write(voiceover_content)

    # Generate image file names based on the timestamp and the index
    image_filenames = [f"{directory}/image_{timestamp}_{idx}.png" for idx, _ in enumerate(images)]

    # Create video
    image_clips = [mpy.ImageClip(img).set_duration(5) for img in image_filenames]
    video_clip = concatenate_videoclips(image_clips, method="compose")
    video_clip = video_clip.set_audio(mpy.AudioFileClip(voiceover_filename))

    video_filename = (
        f"output_video_{timestamp}.mp4"  # The filename already includes a timestamp
    )
    video_clip.write_videofile(video_filename, codec="libx264", fps=24)



```