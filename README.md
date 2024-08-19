# studypal
### Have a conversation about any article on the web

studypal is a fast conversational ai built using [Daily](https://www.daily.co/) to exchange real-time audio between the agent and human, and [Cartesia](https://cartesia.ai) for text-to-speech of the agent. Everything (VAD -> STT -> LLM -> TTS) is orchestrated together using [Pipecat](https://www.pipecat.ai/). 

## Setup

1. Clone the repository
2. Copy `.env.example` to a `.env` file and add API keys 
3. Install the required packages: `pip install -r requirements.txt` 
4. Run `python3 studypal.py` from your command line. 
5. While the app is running, go to the `https://<yourdomain>.daily.co/<room_url>` set in `DAILY_SAMPLE_ROOM_URL` and talk to studypal!
