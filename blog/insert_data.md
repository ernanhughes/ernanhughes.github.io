Certainly! Below is a Python function that uses the `sqlite3` library to insert video data into the `VIDEO_DATA` table. The function assumes you have a SQLite database already set up and connected.

### Function to Insert Video Data

```python
import sqlite3
from typing import Dict
from dataclasses import asdict

def insert_video_data(db_connection: sqlite3.Connection, video_data: VideoInfoData):
    """
    Inserts video data into the VIDEO_DATA table.
    
    :param db_connection: sqlite3.Connection object connected to the database.
    :param video_data: VideoInfoData object containing the video information to insert.
    """
    cursor = db_connection.cursor()
    
    # Convert the VideoInfoData dataclass to a dictionary
    video_dict = asdict(video_data)
    
    # Prepare the SQL statement
    sql = """
    INSERT INTO VIDEO_DATA (
        video_id, title, upload_date, duration, description, genre, 
        is_paid, is_unlisted, is_family_friendly, channel_id, 
        views, likes, dislikes, regionsAllowed, thumbnail_url
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    
    # Prepare the data for insertion
    data = (
        video_dict['id'], 
        video_dict['title'], 
        video_dict['upload_date'], 
        video_dict['duration'], 
        video_dict['description'], 
        video_dict['genre'], 
        int(video_dict['is_paid']), 
        int(video_dict['is_unlisted']), 
        int(video_dict['is_family_friendly']), 
        video_dict['channel_id'], 
        video_dict['views'], 
        video_dict['likes'], 
        video_dict['dislikes'], 
        video_dict['regionsAllowed'], 
        video_dict['thumbnail_url']
    )
    
    try:
        # Execute the SQL statement
        cursor.execute(sql, data)
        
        # Commit the transaction
        db_connection.commit()
        
        print("Video data inserted successfully.")
    except sqlite3.Error as e:
        print(f"An error occurred: {e}")
        db_connection.rollback()

# Example usage:
# Assuming you have a SQLite connection object `conn` and a `VideoInfoData` object `video_info`
# insert_video_data(conn, video_info)
```

### Explanation:
1. **`insert_video_data` Function**:
   - Takes a `sqlite3.Connection` object and a `VideoInfoData` object as inputs.
   - Converts the `VideoInfoData` object to a dictionary using the `asdict` function from the `dataclasses` module.

2. **SQL Statement**:
   - The SQL `INSERT INTO` statement is prepared to match the fields defined in your `VIDEO_DATA` table.
   - The `VALUES` part of the SQL statement contains placeholders (`?`) for binding the data securely to prevent SQL injection.

3. **Data Preparation**:
   - The data to be inserted is prepared as a tuple, matching the order of the fields in the SQL statement.
   - Boolean values (`is_paid`, `is_unlisted`, `is_family_friendly`) are converted to integers (0 or 1) because SQLite doesn't have a boolean data type.

4. **Execution and Error Handling**:
   - The `execute` method of the cursor is used to run the SQL statement with the data.
   - If the operation is successful, the transaction is committed.
   - If an error occurs, the transaction is rolled back, and the error message is printed.

### Usage:
- You would call this function after scraping the video data to insert it into the database.
- Ensure you have a valid SQLite connection object before calling the function.

This approach should handle the insertion of video data into the `VIDEO_DATA` table efficiently.