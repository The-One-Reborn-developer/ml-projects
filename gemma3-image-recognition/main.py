import os

from dotenv import load_dotenv
from huggingface_hub import login


load_dotenv()
login(os.getenv('HUGGINGFACE_TOKEN'))


