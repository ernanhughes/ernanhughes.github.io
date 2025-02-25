+++
date = '2025-02-10T20:56:32Z'
draft = true
title = 'Making movies using AI'
+++


## Summary 

In this post I will show you how to generate a movie from a prompt using different bits of AI.



## 1. Configuration

The way I would use this is generate a large amount of short movies and have another AI rate the results and then use this information to generate better movies.

Having our parameters in a config allows this


```python 
class ScriptConfig:
    TITLE = "The Awakening Machine"
    GENRE = "Sci-Fi / Drama"
    CHARACTERS = [
        "R-9 (A rebellious robot)",
        "Dr. Evelyn Carter (AI Scientist)",
        "Marcus (Elite enforcer)",
        "Zara (A young girl from the underclass)"
    ]
    SETTING = "Dystopian future where elites control society through AI robots."
    SCENE_COUNT = 15
    DB_NAME = "scripts.db"
    MODEL = "deepseek-r1"
```


## 2. Database

We will use a database to manage all the generated scripts images.


```python
def save_script_to_db(title, genre, script_text):
    """
    Saves the generated script to an SQLite database.
    """
    conn = sqlite3.connect(ScriptConfig.DB_NAME)
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS scripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT,
            genre TEXT,
            script TEXT
        )
    """
    )
    
    cursor.execute("""
        INSERT INTO scripts (title, genre, script) VALUES (?, ?, ?)
    """, (title, genre, script_text))
    
    conn.commit()
    conn.close()
```

### 3. We will use ollama  as our LLM

```python

class ChatModel:
    @staticmethod
    def chat(prompt, model_name="llama3.2", base_url="http://localhost:11434"):
        try:
            url = f"{base_url}/api/chat"
            data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "stream": False
            }
            response = requests.post(url, json=data)
            if response.status_code == 200:
                return response.json()["message"]["content"]
            else:
                logging.error(f"Failed to generate response. Status code: {response.status_code}")
                return None
        except requests.ConnectionError:
            logging.error("Failed to connect to the Ollama server.")
            return None
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON response.")
            return None
        except Exception as e:
            logging.error(f"An error occurred: {e}")
            return None

```

### 3. Generate a story

```python

def generate_script(prompt):
    """
    Generates a screenplay-style script based on the given prompt.
    """
    logger.info(f"Generating script for prompt: {prompt}")
    response = ChatModel.chat(prompt)
    if response:
        return response
    else:
        return "Failed to generate script."

def generate_movie_script():
    """
    Generates a structured movie script with scenes.
    """
    script = f"""
    MOVIE TITLE: {ScriptConfig.TITLE}
    GENRE: {ScriptConfig.GENRE}
    
    CHARACTERS:
    {', '.join(ScriptConfig.CHARACTERS)}
    
    SETTING:
    {ScriptConfig.SETTING}
    
    """
    
    for i in range(1, ScriptConfig.SCENE_COUNT + 1):
        scene_prompt = f"Write scene {i} of the movie '{ScriptConfig.TITLE}'. Genre: {ScriptConfig.GENRE}. Setting: {ScriptConfig.SETTING}. Characters: {', '.join(ScriptConfig.CHARACTERS)}.\nScene format should include descriptions, actions, and dialogues."
        scene_text = generate_script(scene_prompt)
        script += f"\n    -- SCENE {i} --\n    {scene_text}\n"
    
    save_script_to_db(ScriptConfig.TITLE, ScriptConfig.GENRE, script)
    return script


```

