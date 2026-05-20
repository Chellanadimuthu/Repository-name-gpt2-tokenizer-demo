import os

from langchain.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers.audio import FasterWhisperParser
from langchain.document_loaders.blob_loaders.youtube_audio import YoutubeAudioLoader

from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma

from config import config


def load_youtube_content(url, save_dir):

    try:

        loader = GenericLoader(
            YoutubeAudioLoader([url], save_dir),
            FasterWhisperParser()
        )

        return loader.load()

    except Exception as e:

        print(e)

        return []


def load_pdf_content(pdf_directory):

    if not os.path.exists(pdf_directory):

        return []

    all_docs = []

    for filename in os.listdir(pdf_directory):

        if filename.endswith(".pdf"):

            filepath = os.path.join(
                pdf_directory,
                filename
            )

            try:

                loader = PyPDFLoader(filepath)

                pages = loader.load()

                all_docs.extend(pages)

            except Exception as e:

                print(e)

    return all_docs


def ingest_all_documents(
    youtube_url,
    youtube_save_dir,
    pdf_directory,
    persist_directory
):

    youtube_docs = load_youtube_content(
        youtube_url,
        youtube_save_dir
    )

    pdf_docs = load_pdf_content(
        pdf_directory
    )

    documents = youtube_docs + pdf_docs

    if not documents:

        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP
    )

    chunks = splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL_NAME
    )

    os.makedirs(
        persist_directory,
        exist_ok=True
    )

    vectordb = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory
    )

    vectordb.persist()

    print("Chroma DB Created")


if __name__ == "__main__":

    ingest_all_documents(
        youtube_url=config.YOUTUBE_VIDEO_URL,
        youtube_save_dir=config.YOUTUBE_AUDIO_SAVE_DIRECTORY,
        pdf_directory=config.PDF_SOURCE_DIRECTORY,
        persist_directory=config.CHROMA_PERSIST_DIRECTORY
    )