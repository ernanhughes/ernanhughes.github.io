+++
date = '2025-02-13T14:22:06Z'
draft = false
title = 'Self-Learning LLMs for Stock Forecasting: A Python Implementation with Direct Preference Optimization'
tags= ['trading', 'DPO', 'Trading', 'Self-Learning']
categories=['trading', 'DPO', 'stock-forecasting']

+++

## Summary

Forecasting future events is a critical task in fields like finance, politics, and technology. However, improving the forecasting abilities of large language models (LLMs) often requires extensive human supervision. 
In this post, we explore a novel approach from the paper [LLMs Can Teach Themselves to Better Predict the Future](https://arxiv.org/abs/2502.05253) that enables LLMs to teach themselves better forecasting skills using self-play and Direct Preference Optimization (DPO). We'll walk through a Python implementation of this method, step by step.

If we can give our applications the ability to auto-tune themselves they would simulate living things. This would be a concrete step towards Artificial General Intelligence (AGI).

---


## **What is Direct Preference Optimization (DPO)?**

**Direct Preference Optimization (DPO) simplifies the fine-tuning of large language models (LLMs) by directly learning from human preferences.** 

Think of DPO as a teacher grading student essays. Instead of giving a detailed rubric (like RLHF), the teacher simply points out which essay is better. The student (the LLM) learns directly from these comparisons, improving over time.

Instead of training a separate reward model (as in Reinforcement Learning from Human Feedback, or RLHF), DPO directly optimizes the LLM itself. This involves presenting the model with pairs of responses and asking humans to select the preferred option. By learning from these preferences, the model is gradually refined to generate outputs that better align with human expectations. 

This approach eliminates the need for a complex reward model, making the fine-tuning process more streamlined and potentially more stable.

In this blog post we take it to the next level and use real time data to determine the preference.

---

### **How DPO Works**
DPO learns from a dataset of **(prompt, preferred response, undesirable response)** triplets. The core idea is to adjust the model so that it **assigns higher probability** to preferred responses and **lower probability** to undesirable ones.

### **Steps in DPO:**
1. **Collect preference data**:  
   - Given a prompt, human labelers (or AI-assisted systems) rank multiple model-generated responses.
   - This results in **preference pairs**: one response is preferred, and another is undesirable.

2. **Optimize the model directly**:  
   - DPO **adjusts the model's logits** (probabilities) to prefer responses that human raters liked.
   - It avoids **reward models and reinforcement learning**, relying purely on likelihood ratio optimization.

3. **Fine-tune with supervised learning**:  
   - The model is fine-tuned by maximizing the likelihood of preferred responses over undesirable ones.

The key advantage is that DPO **sidesteps the complexity of reinforcement learning** while still aligning models with human preferences.

---

### **How DPO Differs from RLHF**
| Feature               | RLHF                                   | DPO |
|-----------------------|--------------------------------------|----------------|
| **Uses reward model?** | ✅ Yes (requires training a separate model) | ❌ No |
| **Uses reinforcement learning?** | ✅ Yes (PPO or other RL methods) | ❌ No |
| **Computationally expensive?** | 🔥 High | ⚡ Lower |
| **Stability** | 🚧 Can be unstable due to RL dynamics | ✅ More stable |
| **Ease of implementation** | 🚀 Complex | ✨ Simple |

---

### **Mathematical Formulation of DPO**
DPO works by **re-weighting the model’s output distribution** so that the probability of the preferred response is higher than the undesirable response.

Given:
- A model **π(θ)** parameterized by **θ**
- A dataset of **(prompt, preferred response, undesirable response)** pairs

DPO optimizes the **logit difference** using the objective:

\[
L(\theta) = \sum_{(x, y^+, y^-)} \log \frac{\pi_\theta(y^+ | x)}{\pi_\theta(y^- | x)}
\]

This ensures that the model assigns higher probability to **preferred responses** without needing a separate reward model.

---

### **Why Use DPO?**
🔹 **Eliminates complexity**: No need for **reward models** or **RL algorithms**.  
🔹 **More stable**: Avoids **high variance and instability** seen in RLHF.  
🔹 **Faster & cheaper**: Requires **less computation** than RL-based approaches.  
🔹 **Easier to implement**: Uses **standard supervised fine-tuning techniques** instead of reinforcement learning.  

---



## **1. Forecasting Dataset**

### Training Data

The first step is to gather a dataset of **binary outcome forecasting questions**. I took a really simple approach.

* **1. We get news for stocks.**
* **2. Is the news positive for the stock.**
* **3. If this is the case the stock will appreciate**

I know this is simplistic and not always true but generally or most ot the time it is the case.

For this post we are using a dataset from huggingface [oliverwang15/us_stock_news_with_price](https://huggingface.co/datasets/oliverwang15/us_stock_news_with_price)

This dataset has the following format

#### Data Description
* **date**: The date of the news published.
* **stock**: The symbol of the stocks the news related to. (checked by whether title or content has the company information.
* **title**: The title of the news.
* **content**: The content of the news.
* **trading_date**: Here is the assumed trading date, which should be the next date of the publish date.
* **exact_trading_date**: The exact next trading date after the news was made public.
* **ts_{-30...-1}**: Stock prices before the exact trading date. (30 trading days)
* **ts_0**: Stock prices of the exact trading date.
* **ts_{1...15}**: Stock prices after the exact trading date. (15trading days)


For now we are only interested in a subset of the columns

| Column  | New Name | Description |
| ----- | ----- | ----- |
|stock|ticker|Stock ticker|
|title|news_title|Title of the news story|
|content|news_summary|The actual news content|
|ts_0|next_day_price|The price after the news|
|ts_-1|news_day_price|The price before the news|
|     |   | Calculated sentiment| 



```python

def get_data():
    try:
         # Load dataset from Hugging Face
        dataset = load_dataset("oliverwang15/us_stock_news_with_price")
        df = dataset["train"].to_pandas()
        print("Available Columns:", df.columns)

        # Extract necessary columns and drop missing values
        df = df[["stock", "title", "content", "ts_0", "ts_-1", "trading_date", "exact_trading_date"]].dropna()

        # Calculate news effect: positive if next_day_price >= news_day_price, else negative
        df["news_effect"] = df.apply(
            lambda row: "positive" if row["ts_0"] >= row["ts_-1"] else "negative", axis=1
        )

        # Rename columns for clarity
        df_news = df.rename(
            {
                "stock": "ticker",
                "title": "news_title",
                "content": "news_summary",
                "ts_0": "next_day_price",
                "ts_-1": "news_day_price",
            },
            axis=1,
        )
        print("Final Columns:", df_news.columns)
        return df_news
    except Exception as e:
        print(f"Error loading or processing data: {e}")
        return None
```

This dataset contains forecasting questions along with binary outcomes (1 for Yes, 0 for No).

## 2. Fetching Relevant News
The paper enhances the forecasting ability of LLMs by providing news summaries as additional context. 

We have the news date in our dataset

Python Code: Generating News Summaries lets summarize it and get the sentiment

### Create a database to store our data

Creating three tables

* **stock_news**: Essentially this table is the information we are interested from initial data source.
* **news_sentiment**: This contains the generated sentiment from the model  (positive/negative)
*  **news_forecast**: this table is for the forecast from self play
*  **model_updates**: the idea is that the model will be updated onn a schedule with the latest news. This table is to track that process.


```python
def create_database():
    # Create SQLite database connection
    conn = sqlite3.connect("stock_news.db")
    cursor = conn.cursor()

    # Enable foreign key support
    cursor.execute("PRAGMA foreign_keys = ON;")

    # Create tables in SQLite
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS stock_news (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT,
            news_title TEXT,
            news_summary TEXT,
            next_day_price FLOAT,
            news_day_price FLOAT,
            trading_date TEXT,
            exact_trading_date TEXT,
            news_effect TEXT
        );
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS news_sentiment (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id INTEGER,
            sentiment TEXT,
            news_explanation TEXT,
            FOREIGN KEY (news_id) REFERENCES stock_news(id) ON DELETE CASCADE
        );
    """
    )

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS news_forecast (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            news_id INTEGER,
            ticker TEXT,
            forecast_1 TEXT,
            probability_1 FLOAT,
            forecast_2 TEXT,
            probability_2 FLOAT,
            best_forecast TEXT,
            FOREIGN KEY (news_id) REFERENCES news_sentiment(news_id) ON DELETE CASCADE
        );
    """
    )
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_updates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            update_date TEXT,
            brier_score FLOAT
        );
    """)

    ## insert the hugging face data
    def insert_data(df_data):
        conn = sqlite3.connect("stock_news.db")
        cursor = conn.cursor()
        for _, row in df_data.iterrows():
            cursor.execute("""
                INSERT INTO stock_news (ticker, news_title, news_summary, next_day_price, news_day_price, news_effect)
                VALUES (?, ?, ?, ?, ?, ?);
            """, (row["ticker"], row["news_title"], row["news_summary"], row["next_day_price"],
                    row["news_day_price"], row["news_effect"]))
        conn.commit()
        conn.close()

```

### Analyze sentiment with explanation

Here we ask the model to determine the sentiment of the news. We also ask it to generate a brief explanation on why it thinks this is the case.

Sentiment analysis helps us understand whether news is likely to have a positive or negative impact on stock prices. By analyzing the sentiment of news articles, we aim to better predict stock movements.

```python
def analyze_sentiment_and_explain(news_title, news_summary):
    """Use the model to analyze sentiment and provide an explanation in JSON format."""
    prompt = f"""
    Analyze the sentiment of the following stock news article and explain your analysis

    Title: {news_title}
    Summary: {news_summary}

    Provide the response in **valid JSON format**:
    {{
        "sentiment": "<positive/negative/neutral>",
        "explanation": "<Brief explanation>"
    }}
    """
    response: ChatResponse = chat(model='qwen2.5', messages=[
        {
            'role': 'user',
            'content': f'{prompt}'
        }])


    response_text = response['message']['content'].strip()

    try:
        sentiment_data = json.loads(response_text)  # Parse JSON response
        sentiment = sentiment_data.get(
            "sentiment", "neutral"
        )  # Default to Neutral if sentiment is not provided
        sentiment = sentiment.lower()
        explanation = sentiment_data.get("explanation", "No explanation available.")
    except json.JSONDecodeError as e:
        # If JSON parsing fails, default to Neutral sentiment
        sentiment, explanation = "neutral", f"Sentiment could not be determined. {str(e)}"

    return sentiment, explanation

def get_sentiment():
    conn = sqlite3.connect("stock_news.db")
    cursor = conn.cursor()
    cursor.execute("""SELECT id, ticker, news_title, news_summary
                        FROM stock_news
                        LIMIT 20;""") # just a subset of the data for now
    news_data = cursor.fetchall()

    for news_id, ticker, title, summary in news_data:
        # call teh model with our data
        sentiment, explanation = analyze_sentiment_and_explain(title, summary)

        # Insert sentiment & explanation into news_sentiment table
        cursor.execute(
            """
            INSERT INTO news_sentiment (news_id, sentiment, news_explanation)
            VALUES (?, ?, ?);
        """,
            (news_id, sentiment, explanation),
        )
    conn.commit()
    conn.close()
```

Example results:

|id|news_id|sentiment|news_explanation|
|----|-----|-----|----|
|1|1|positive|The article is positive as it mentions that RadioShack should post 'outsized gains next year' according to Barclays. It also lists several positive factors contributing to the outlook, such as the addition of T-Mobile as a carrier and new branding campaigns. The conclusion that the stock 'is a solid investment for 2010' further reinforces the positive sentiment.|
|2|2|neutral|The article presents both positive and negative aspects of AT&T's network upgrade efforts. On one hand, it acknowledges the company’s response to criticism by upgrading as fast as possible and planning measures to handle high-bandwidth users. However, it also highlights that despite this effort, financial data shows that AT&T has spent less on network buildout every quarter since the iPhone was launched, which could be seen as a negative sign for future network quality or expansion. The overall tone is balanced without strong positive or negative connotations.|


---

## 3. Generating Forecasts via Model Self-Play
The core innovation in the paper is self-play, where the model generates multiple forecasts for the same question. This allows the model to explore different reasoning paths for the same event.

Here we are saying to the model, this is what `Bill` said what do you think are the odds. This force the model to think deeper on the problem.
It also give us away to measure how good the model has gotten predicting stock movement from the news. 

Now know how the model is that the news is positive or negative. If we feed this in to the model getting it to improve these scores over time it should improve. 

```python

def extract_json_from_tags(text):
    """Extract JSON content between ```json and ``` tags."""
    match = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    return match.group(1) if match else None


def forecast_sentiment(ticker, sentiment, news_explanation):
    """Prompt GPT-4 to generate a stock movement forecast as 'positive' or 'negative'."""
    prompt = f"""
    Given the question: Is this forecast {sentiment} for stock {ticker}.
    And the following news summary about the stock: {news_explanation}

    Provide a JSON response with the following:
    {{
        "forecast": "<positive/negative>",
        "probability": <A number between 0 and 1>
        "explanation": "<Brief explanation>"
    }}
    """



    response: ChatResponse = chat(model='qwen2.5', messages=[
        {
            'role': 'user',
            'content': f'{prompt}'
        }])

    response_text = response['message']['content'].strip()
    if "```json" in response_text:
        response_text = extract_json_from_tags(response_text)

    try:
        forecast_data = json.loads(response_text)  # Parse JSON response
        forecast = forecast_data.get(
            "forecast", "missing"
        )  # Default to "missing" if missing
        probability = forecast_data.get("probability", 0.0)
    except json.JSONDecodeError:
        forecast, probability = "missing", 0.0

    return forecast, probability

    # Fetch sentiment data for forecasting along with the stock ticker


def gen_forecast():
    conn = sqlite3.connect("stock_news.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT news_sentiment.news_id,
        stock_news.ticker, 
        news_sentiment.sentiment, 
        news_sentiment.news_explanation 
        FROM news_sentiment
        JOIN stock_news ON news_sentiment.news_id = stock_news.id;
    """
    )
    sentiment_data = cursor.fetchall()
    for news_id, ticker, sentiment, explanation in sentiment_data:
        print(f"Generating forecasts for id: {news_id} stock {ticker} with sentiment {sentiment}...")
        forecast_1, probability_1 = forecast_sentiment(
            ticker, "positive", explanation
        )
        forecast_2, probability_2 = forecast_sentiment(
            ticker, "negative", explanation
        )

        if probability_1 > probability_2:
            best_forecast = forecast_1
        else:
            best_forecast = forecast_2

        # Insert forecasts into news_forecast table
        cursor.execute(
            """
            INSERT INTO news_forecast (news_id, ticker, forecast_1, probability_1, forecast_2, probability_2, best_forecast)
            VALUES (?, ?, ?, ?, ?, ?, ?);
        """,
            (news_id, ticker, forecast_1, probability_1, forecast_2, probability_2, best_forecast),
        )

    # Commit forecasts to database
    conn.commit()
    conn.close()

```

Example results

|id|news_id|ticker|forecast_1|probability_1|forecast_2|probability_2|best_forecast|
| ---- | ---- | ----- | ----- | ---- | ---- | ----- | ----- |
|1|1|AAPL|positive|0.95|negative|0.2|positive|
|2|2|AAPL|negative|0.5|negative|0.4|negative|
|3|3|AAPL|negative|0.8|negative|0.8|negative|



## 4. Ranking Forecasts Based on Accuracy
Once real-world outcomes are available, we can rank forecasts based on their proximity to the true outcome. 

Ranking forecasts based on their accuracy helps us identify which predictions are most reliable. This ensures that we prioritize the best forecasts for decision-making.


The **ranking function** used in the paper is:

\[
r(p, o) = |p - o|
\]

where:
- \( p \) is the predicted probability
- \( o \) is the actual outcome (0 or 1)

Python Code: Ranking Forecasts

```python

def ranking_metric(prediction, actual_outcome):
    """Calculate the absolute error between prediction and actual outcome."""
    return abs(prediction - actual_outcome)

# Convert forecast strings to float probabilities
df["forecast_1"] = df["forecast_1"].str.extract(r'(\d\.\d+)').astype(float)
df["forecast_2"] = df["forecast_2"].str.extract(r'(\d\.\d+)').astype(float)

# Compute ranking scores
df["rank_1"] = df.apply(lambda row: ranking_metric(row["forecast_1"], row["outcome"]), axis=1)
df["rank_2"] = df.apply(lambda row: ranking_metric(row["forecast_2"], row["outcome"]), axis=1)

# Determine the better forecast
df["better_forecast"] = df.apply(lambda row: "Forecast 1" if row["rank_1"] < row["rank_2"] else "Forecast 2", axis=1)

# Display dataset with rankings
tools.display_dataframe_to_user(name="Ranked Forecasts", dataframe=df)

```

This step identifies which forecast was closer to the actual outcome.

---

## 5. Fine-Tuning with Direct Preference Optimization (DPO)
The paper fine-tunes the model by using pairs of ranked reasoning traces. We'll prepare a fine-tuning dataset based on our rankings.


Python Code: Preparing Fine-Tuning Data

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType

def fine_tune_dpo(df):
    model_name = "mistralai/Mistral-7B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"]
    )
    model = get_peft_model(model, peft_config)
    
    train_texts = [f"News: {row['content']}\nSentiment: {row['news_effect']}\n" for _, row in df.iterrows()]
    train_labels = [row['news_effect'] for _, row in df.iterrows()]
    
    tokenized_data = tokenizer(train_texts, padding=True, truncation=True, return_tensors="pt")
    
    training_args = TrainingArguments(
        output_dir="./dpo_model",
        per_device_train_batch_size=4,
        num_train_epochs=3,
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_data,
    )
    trainer.train()
    model.save_pretrained("./dpo_model")
    tokenizer.save_pretrained("./dpo_model")

# Train the model
fine_tune_dpo(df)

```

Check this blog post: [Mastering LLM Fine-Tuning: A Practical Guide with LLaMA-Factory and LoRA]({{< relref "post/fine_tuning.md" >}}) to learn more about fine tuning LLM's.


By integrating Direct Preference Optimization (DPO), we enhance an LLM’s ability to forecast stock movements based on news sentiment. This approach eliminates the need for a complex reward model, making fine-tuning simpler, faster, and more stable.

Additionally, ranking forecasts ensures we prioritize more accurate predictions, further improving decision-making. 🚀


---

## **6. Evaluating the Model: Brier Score Calculation**
The paper evaluates model performance using the **Brier Score**, defined as:

\[
BS = \frac{1}{N} \sum (p_i - o_i)^2
\]

Where:
- \( p_i \) is the predicted probability
- \( o_i \) is the actual outcome

**Python Code: Computing Brier Score**
```python
def brier_score(predictions, outcomes):
    """Compute the Brier Score for probabilistic forecasts."""
    return ((predictions - outcomes) ** 2).mean()

brier = brier_score(df["forecast_1"], df["outcome"])
print(f"Brier Score: {brier:.4f}")
```

A **lower Brier score** indicates **better forecasting performance**.

---