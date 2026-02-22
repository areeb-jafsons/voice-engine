import asyncio
import time
import soundfile as sf
from deepgram import AsyncDeepgramClient
from deepgram.core.events import EventType

DEEPGRAM_API_KEY = "0aef0bb379bd5e396104ed65b3389c26ea969e14" # Using the provided Bearer token earlier
AUDIO_FILE = "Recording.wav"

async def main():
    client = AsyncDeepgramClient(DEEPGRAM_API_KEY)

    with sf.SoundFile(AUDIO_FILE) as f:
        audio_data = f.read(dtype="int16")

    first_transcript = None
    end_of_turn = None

    start_time = time.perf_counter()

    async with client.listen.v2.connect(
        model="flux-general-en",
        encoding="linear16",
        sample_rate="16000",
        eot_threshold=0.7,
        eager_eot_threshold=0.5
    ) as connection:

        async def on_message(message):
            nonlocal first_transcript, end_of_turn

            now = time.perf_counter()

            if hasattr(message, "transcript") and message.transcript:
                if first_transcript is None:
                    first_transcript = now
                    print(f"FIRST_TRANSCRIPT_MS={(now-start_time)*1000:.2f}")

            if getattr(message, "type", None) == "EndOfTurn":
                end_of_turn = now
                print(f"END_OF_TURN_MS={(now-start_time)*1000:.2f}")

        connection.on(EventType.MESSAGE, on_message)

        await connection.start_listening()

        # 80ms chunks (recommended by Deepgram)
        chunk_bytes = 2560  # 80ms @ 16kHz linear16
        step = chunk_bytes // 2

        for i in range(0, len(audio_data), step):
            chunk = audio_data[i:i+step]
            await connection._send(chunk.tobytes())
            await asyncio.sleep(0.08)

        await asyncio.sleep(2)

asyncio.run(main())
