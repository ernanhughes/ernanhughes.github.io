import os
import markdown
import requests
from bs4 import BeautifulSoup

class Config:
    """Configuration class for setting default values."""
    DIRECTORY = "./docs"
    TITLE = "Project Documentation"
    BASE_CONTENT = "This documentation provides an index of all Markdown files, organized by subfolder and sorted by the latest modified date."
    GITHUB_BASE_URL = "https://github.com/yourusername/yourrepo/blob/main"  # Change this to your repo
    SITE_SEARCH_URL = "https://programmer.ie/search?q="  # Programmer.ie search URL
    EXCLUDED_FILES = ["README.md"]  # Files to exclude from the generated table

class ReadmeGenerator:
    def __init__(self, config=Config):
        """Initialize with a config class."""
        self.config = config

    def get_markdown_files_by_folder(self):
        """Recursively finds all markdown files grouped by subfolder and sorts them by last modified date."""
        md_files = {}

        for root, _, files in os.walk(self.config.DIRECTORY):
            md_files[root] = [
                (f, os.path.getmtime(os.path.join(root, f)))
                for f in files if f.endswith(".md") and f not in self.config.EXCLUDED_FILES
            ]
        
        # Sort each folder's files by modification date (newest first)
        for folder in md_files:
            md_files[folder] = sorted(md_files[folder], key=lambda x: x[1], reverse=True)

        return md_files

    def get_brief_content(self, filepath):
        """Extracts a brief summary from the first paragraph of the Markdown file."""
        with open(filepath, "r", encoding="utf-8") as f:
            md_content = f.read()

        html = markdown.markdown(md_content)
        soup = BeautifulSoup(html, "html.parser")
        paragraphs = soup.find_all("p")
        
        return paragraphs[0].text[:200] + "..." if paragraphs else "No summary available."

    def search_programmer_ie(self, filename):
        """Searches programmer.ie for a related post."""
        search_url = self.config.SITE_SEARCH_URL + filename.replace(".md", "")
        try:
            response = requests.get(search_url, timeout=5)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                first_result = soup.find("a")  # Finds the first link in the search results
                
                if first_result and "href" in first_result.attrs:
                    return first_result["href"]
        except requests.RequestException:
            pass
        return ""

    def generate_table_for_folder(self, folder, files):
        """Creates a Markdown table for a specific folder."""
        table = "| File | Summary | Related Post |\n|------|---------|--------------|\n"
        
        for filename, _ in files:
            filepath = os.path.join(folder, filename)
            file_url = f"{self.config.GITHUB_BASE_URL}/{os.path.relpath(filepath, self.config.DIRECTORY)}"
            summary = self.get_brief_content(filepath)
            related_post = self.search_programmer_ie(filename)

            table += f"| [{filename}]({file_url}) | {summary} | [{related_post}]({related_post}) |\n" if related_post else f"| [{filename}]({file_url}) | {summary} | |\n"

        return table

    def generate_readme(self):
        """Generates the final README.md content with sections for each subfolder."""
        content = f"# {self.config.TITLE}\n\n{self.config.BASE_CONTENT}\n\n"

        files_by_folder = self.get_markdown_files_by_folder()

        for folder, files in files_by_folder.items():
            if files:
                folder_name = os.path.relpath(folder, self.config.DIRECTORY) or "Root Directory"
                content += f"## {folder_name}\n\n{self.generate_table_for_folder(folder, files)}\n\n"

        readme_path = os.path.join(self.config.DIRECTORY, "README.md")

        with open(readme_path, "w", encoding="utf-8") as f:
            f.write(content)

        print(f"✅ README.md generated at: {readme_path}")

# Usage Example
if __name__ == "__main__":
    generator = ReadmeGenerator(config=Config)
    generator.generate_readme()
