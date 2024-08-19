import aiohttp
import asyncio
import os
import sys
import requests
import io
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
import tiktoken

from pipecat.frames.frames import LLMMessagesFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.llm_response import (
    LLMAssistantResponseAggregator, LLMUserResponseAggregator)
from pipecat.services.cartesia import CartesiaTTSService
from pipecat.services.openai import OpenAILLMService
from pipecat.transports.services.daily import DailyParams, DailyTransport
from pipecat.vad.silero import SileroVADAnalyzer

from runner import configure

from loguru import logger

from dotenv import load_dotenv
load_dotenv(override=True)

# Run this script directly from your command line. 
# This project was adapted from https://github.com/pipecat-ai/pipecat/blob/main/examples/foundational/07d-interruptible-cartesia.py

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

def get_article_content(url):
    if 'arxiv.org' in url:
        return get_arxiv_content(url)
    else:
        return get_wikipedia_content(url)

# Helper function to extract content from Wikipedia url (this is technically agnostic to URL type but will work best with Wikipedia articles)
def get_wikipedia_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    
    content = soup.find('div', {'class': 'mw-parser-output'})
    
    if content:
        return content.get_text()
    else:
        return "Failed to extract Wikipedia article content."

# Helper function to extract content from arXiv url 
def get_arxiv_content(url):
    if '/abs/' in url:
        url = url.replace('/abs/', '/pdf/')
    if not url.endswith('.pdf'):
        url += '.pdf'

    response = requests.get(url)
    if response.status_code == 200:
        pdf_file = io.BytesIO(response.content)
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text
    else:
        return "Failed to download arXiv PDF."

# Count number of tokens used in model and truncate the content 
def truncate_content(content, max_tokens=7000):
    encoding = tiktoken.encoding_for_model("gpt-4")
    tokens = encoding.encode(content)
    if len(tokens) > max_tokens:
        truncated_tokens = tokens[:max_tokens]
        return encoding.decode(truncated_tokens)
    return content

# This is the main function that handles STT -> LLM -> TTS 
async def main():
    url = input("Enter the URL of the article you would like to talk about: ")
    article_content = get_article_content(url)
    article_content = truncate_content(article_content)

    async with aiohttp.ClientSession() as session:
        (room_url, token) = await configure(session)

        transport = DailyTransport(
            room_url,
            token,
            "studypal",
            DailyParams(
                audio_out_sample_rate=44100,
                audio_out_enabled=True,
                transcription_enabled=True,
                vad_enabled=True,
                vad_analyzer=SileroVADAnalyzer()
            )
        )

        tts = CartesiaTTSService(
            api_key=os.getenv("CARTESIA_API_KEY"),
            voice_id="a0e99841-438c-4a64-b679-ae501e7d6091",  # Barbershop Man
            sample_rate=44100,
        )

        llm = OpenAILLMService(
            api_key=os.getenv("OPENAI_API_KEY"),
            model="gpt-4")

        messages = [
            {
                "role": "system",
                "content": f"""You are an AI study partner. You have been given the following article content:

{article_content}

Your task is to help the user concisely understand and learn from this article. THESE RESPONSES SHOULD BE ONLY 1-3 SENTENCES AND CONCISE. THIS INSTRUCTION IS VERY IMPORTANT. RESPONSES SHOULDN'T BE LONG.
""",
            },
        ]

        tma_in = LLMUserResponseAggregator(messages)
        tma_out = LLMAssistantResponseAggregator(messages)

        pipeline = Pipeline([
            transport.input(),
            tma_in,
            llm,
            tts,
            tma_out,
            transport.output(),
        ])

        task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True, enable_metrics=True))

        @transport.event_handler("on_first_participant_joined")
        async def on_first_participant_joined(transport, participant):
            transport.capture_participant_transcription(participant["id"])
            messages.append(
                {"role": "system", "content": "Hello! I'm ready to discuss the article with you. What would you like to learn about?"})
            await task.queue_frames([LLMMessagesFrame(messages)])

        runner = PipelineRunner()

        await runner.run(task)

if __name__ == "__main__":
    asyncio.run(main())