import os

class Config:

    YOUTUBE_VIDEO_URL = "https://www.youtube.com/watch?v=uFhDGagZzjs"

    YOUTUBE_AUDIO_SAVE_DIRECTORY = "docs/youtube/"

    PDF_SOURCE_DIRECTORY = "data"

    CHROMA_PERSIST_DIRECTORY = "docs/chroma"

    EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-large"

    CHUNK_SIZE = 2028

    CHUNK_OVERLAP = 250

    def __init__(self):

        os.makedirs(
            self.PDF_SOURCE_DIRECTORY,
            exist_ok=True
        )

config = Config()