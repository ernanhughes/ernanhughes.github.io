Certainly! Splitting the `scrape_video_data` function into smaller, more focused functions will improve readability and make the code easier to maintain. Here's how you could refactor the function:

```python
from urllib.request import urlopen
from rich import print
from bs4 import BeautifulSoup
import re
from dataclasses import dataclass, asdict


@dataclass
class VideoInfoData:
    id: str
    title: str = ''
    url: str = ''
    upload_date: str = ''
    duration: str = ''
    description: str = ''
    genre: str = ''
    is_paid: bool = False
    is_unlisted: bool = False
    is_family_friendly: bool = False
    channel_id: str = ''
    views: int = 0
    likes: int = 0
    dislikes: int = 0
    thumbnail_url: str = ''
    playerType: str = ''
    regionsAllowed: str = ''


class VideoInfo:

    def __init__(self, video_id: str):
        self.video_id = video_id
        self.video_data = self.scrape_video_data()

    def is_true(self, val: str) -> bool:
        return val.lower() not in ["false", "0"]

    def remove_comma(self, s: str) -> str:
        return "".join(s.split(","))

    def scrape_video_data(self) -> VideoInfoData:
        url = f"https://www.youtube.com/watch?v={self.video_id}"
        html = urlopen(url).read()
        soup = BeautifulSoup(html, "lxml")
        video = VideoInfoData(id=self.video_id, url=url)

        item_props = soup.find(id="watch7-content")
        if not item_props or len(item_props.contents) <= 1:
            raise MissingIdError(f"Video with the ID {self.video_id} does not exist")

        self.extract_basic_info(item_props, video)
        self.extract_likes_dislikes(soup, video)

        print(asdict(video))  # Print the video data as a dictionary
        return video

    def extract_basic_info(self, item_props, video: VideoInfoData) -> None:
        """Extract basic video info from item properties."""
        for tag in item_props.find_all(itemprop=True, recursive=False):
            key = tag["itemprop"]
            if key == "name":
                video.title = tag["content"]
            elif key == "duration":
                video.duration = tag["content"]
            elif key == "datePublished":
                video.upload_date = tag["content"]
            elif key == "genre":
                video.genre = tag["content"]
            elif key == "paid":
                video.is_paid = self.is_true(tag["content"])
            elif key == "unlisted":
                video.is_unlisted = self.is_true(tag["content"])
            elif key == "isFamilyFriendly":
                video.is_family_friendly = self.is_true(tag["content"])
            elif key == "thumbnailUrl":
                video.thumbnail_url = tag["href"]
            elif key == "interactionCount":
                video.views = int(tag["content"])
            elif key == "channelId":
                video.channel_id = tag["content"]
            elif key == "description":
                video.description = tag["content"]
            elif key == "playerType":
                video.playerType = tag["content"]
            elif key == "regionsAllowed":
                video.regionsAllowed = tag["content"]

    def extract_likes_dislikes(self, soup, video: VideoInfoData) -> None:
        """Extract likes and dislikes from the script tags."""
        all_scripts = soup.find_all("script")
        for script in all_scripts:
            try:
                if script.string and "ytInitialData" in script.string:
                    video.likes = self.extract_stat("LIKE", script.string)
                    video.dislikes = self.extract_stat("DISLIKE", script.string)
            except (AttributeError, IndexError, ValueError) as e:
                print(f"Error parsing like/dislike counts: {e}")

    def extract_stat(self, label: str, script_content: str) -> int:
        """Extract specific statistic (likes/dislikes) from the script content."""
        try:
            match = re.findall(f'label(.*)', re.findall(f'{label}(.*?){label.lower()}', script_content)[0])[0]
            result = ("".join(match.split(",")).split('"')[-1]).strip()
            return int(result)
        except (IndexError, ValueError) as e:
            print(f"Error extracting {label} count: {e}")
            return 0


class MissingIdError(ValueError):
    pass
```

### Breakdown of Changes:

1. **`extract_basic_info` Method**:
   - Handles the extraction of basic information like title, duration, upload date, etc., from the `item_props` section.

2. **`extract_likes_dislikes` Method**:
   - Extracts the likes and dislikes by analyzing the script tags in the HTML content.

3. **`extract_stat` Method**:
   - A helper method used by `extract_likes_dislikes` to handle the specific extraction of likes and dislikes using regex.

### Benefits:
- **Modularity**: Each function now has a clear responsibility, which makes the code easier to understand and maintain.
- **Reusability**: If you need to extract only the basic info or just the like/dislike counts, you can call these methods separately.
- **Improved Readability**: The main `scrape_video_data` function is now shorter and more focused, improving overall readability.

This refactoring maintains the functionality while making the codebase more modular and easier to extend.