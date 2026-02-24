from dotenv import load_dotenv
import os

load_dotenv()  # loads .env from current working directory

print("GitHub token loaded:", "GITHUB_TOKEN" in os.environ)
