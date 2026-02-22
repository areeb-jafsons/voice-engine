import asyncio
import websockets
import json
import time
import soundfile as sf
import numpy as np
import scipy.signal

DEEPGRAM_API_KEY = "0aef0bb379bd5e396104ed65b3389c26ea969e14" # Insert the bearer token again here to test
AUDIO_FILE = "Recording.wav"

WS_URL = (
    "wss://api.deepgram.com/v2/listen?"
    "model=flux-general-en"
    "&encoding=linear16"
    "&sample_rate=16000"
    "&eot_threshold=0.7"
    "&eager_eot_threshold=0.5"
)

async def main():
    headers = {
        "Authorization": f"Token {DEEPGRAM_API_KEY}"
    }

    # Load audio
    data, sr = sf.read(AUDIO_FILE)

    print("Original SR:", sr)
    print("Original dtype:", data.dtype)

    # Convert to mono if needed
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)

    # Resample to 16kHz
    if sr != 16000:
        number_of_samples = round(len(data) * float(16000) / sr)
        data = scipy.signal.resample(data, number_of_samples)

    # Convert float to int16 properly
    data = np.clip(data, -1.0, 1.0)
    audio_data = (data * 32767).astype(np.int16)

    print("Processed dtype:", audio_data.dtype)
    print("Processed max:", np.max(audio_data))

    first_transcript = None
    end_of_turn = None
    connected = False
    speech_start_time = None

    start_time = time.perf_counter()

    async with websockets.connect(WS_URL, extra_headers=headers) as ws:

        async def receiver():
            nonlocal first_transcript, end_of_turn, connected

            async for message in ws:
                msg = json.loads(message)
                now = time.perf_counter()

                print("RAW_EVENT:", msg)

                if msg.get("type") == "Connected":
                    connected = True

                # Flux V2 transcript structure
                if msg.get("type") == "TurnInfo":
                    transcript = msg.get("transcript", "").strip()

                    if transcript and first_transcript is None and speech_start_time is not None:
                        first_transcript = now
                        print(f"FIRST_TRANSCRIPT_MS={(now-speech_start_time)*1000:.2f}")

                    # EndOfTurn detection
                    if msg.get("event") == "End" and speech_start_time is not None:
                        end_of_turn = now
                        print(f"END_OF_TURN_MS={(now-speech_start_time)*1000:.2f}")

        receiver_task = asyncio.create_task(receiver())

        while not connected:
            await asyncio.sleep(0.01)

        chunk_bytes = 2560
        step = chunk_bytes // 2

        for i in range(0, len(audio_data), step):
            if speech_start_time is None:
                speech_start_time = time.perf_counter()
            chunk = audio_data[i:i+step]
            await ws.send(chunk.tobytes())
            await asyncio.sleep(0.08)

        await ws.send(json.dumps({"type": "CloseStream"}))
        await asyncio.sleep(3)

        receiver_task.cancel()

asyncio.run(main())
