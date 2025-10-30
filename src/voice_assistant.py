import asyncio
import base64
import json
import os
import sounddevice as sd
import websockets
from dotenv import load_dotenv
import numpy as np
import sys


load_dotenv()

SAMPLE_RATE = 24000
CHANNELS = 1

#sd.default.device = (2,None)

async def send_audio(ws):
    # <-- grab the loop here, where we *are* on the asyncio thread
    loop = asyncio.get_running_loop()

    def callback(indata, frames, time, status):
        if status:
            print(f"Mic warning: {status}", file=sys.stderr)

        # Convert raw bytes → base64 string
        encoded = base64.b64encode(indata.tobytes()).decode("utf-8")
        payload = {"type": "input_audio_buffer.append",
                   "audio": encoded
        }
        # Schedule send on asyncio loop
        asyncio.run_coroutine_threadsafe(
            ws.send(json.dumps(payload)),
            loop
        )
        
    # Open the input stream
    with sd.InputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        callback=callback,
        dtype="int16"
        ):
        # just keep the coroutine alive
        await asyncio.Future() #run forever
        '''
        while True:
            # wait at least 100ms+ of audio before committing
            await asyncio.sleep(0.05)
            print(f"⏱ Checking commit threshold: {sample_count}/{THRESHOLD}")
            if sample_count >= THRESHOLD:
                print("🔄 Threshold reached—committing buffer now!")
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
                sample_count = 0
        '''

'''
This function will play audio for each chunk it receives from the API. It outputs "chopped" audio. I'll change it to a openstream instead.

async def play_audio(chunk: bytes):
    "Decode war PCM16LE bytes and play"
    audio = np.frombuffer(chunk, dtype=np.int16)
    sd.play(audio, SAMPLE_RATE)
    sd.wait()
'''
async def receive_audio(ws, output_stream):
    "Listen for API events and handle them"
    user_is_sepeaking = False

    async for message in ws:
        event = json.loads(message)
        etype = event.get("type")
        # VAD
        if etype == "input_audio_buffer.speech_started":
            user_is_sepeaking = True
            print("🟢 Speech detected. Listening...")
        # EOU
        elif etype == "input_audio_buffer.speech_stopped":
            if user_is_sepeaking:
                user_is_sepeaking = False
                print("🔴 Speech stopped. Sending audio")
                # give any pending append() calls a chance to be sent
                await asyncio.sleep(0.1)
                print("🔄 Committing buffer now!")
                await ws.send(json.dumps({"type": "input_audio_buffer.commit"}))
        # API Audio
        elif etype == "response.audio.delta":
            # Real-time audio chunk
            chunk = base64.b64decode(event["delta"])
            #await play_audio(chunk)
            audio = np.frombuffer(chunk, dtype=np.int16)
            output_stream.write(audio)
        # Audio transcript
        elif etype == "response.audio_transcript.done":
            print(f"💬 Assistant: {event['transcript']}")
        elif etype == "error":
            err = event.get("error", {})
            # ignore the “empty buffer” error
            if err.get("code") == "input_audio_buffer_commit_empty":
                # print("⚠️ Ignoring empty-buffer commit error.")
                continue
            # for anything else, bail out
            print(f"❌ API error: {err}", file=sys.stderr)
            break
        
async def main():
    api_key = os.getenv("OPENAI_API_KEY")
    url = "wss://api.openai.com/v1/realtime?model=gpt-4o-realtime-preview"
    headers = [
        ("Authorization", f"Bearer {api_key}"),
        ("OpenAI-Beta", "realtime=v1")
    ]
    if not api_key:
        raise ValueError("Missing the OpenAI API key. Please set it in the .env file.")
    
    # Open one continuous output stream
    with sd.OutputStream(
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        dtype="int16"
    ) as output_stream:
        async with websockets.connect(url, additional_headers = headers) as ws:
            # Launch, send and receive tasks concurrently
            send_task = asyncio.create_task(send_audio(ws))
            # pass the output stream into receive_audio
            receive_task = asyncio.create_task(receive_audio(ws, output_stream))
            await asyncio.gather(send_task, receive_task)

if __name__ == "__main__":
    asyncio.run(main())  
