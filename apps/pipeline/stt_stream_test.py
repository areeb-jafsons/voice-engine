import asyncio
import time
import json
import websockets
import soundfile as sf

DEEPGRAM_API_KEY = "0aef0bb379bd5e396104ed65b3389c26ea969e14"
AUDIO_FILE = "test.wav"

async def main():
    url = "wss://api.deepgram.com/v1/listen?model=nova-3&language=en&interim_results=true&endpointing=300&vad_events=true&encoding=linear16&sample_rate=16000"
    headers = {
        "Authorization": f"Bearer {DEEPGRAM_API_KEY}"
    }

    first_interim_time = None
    final_time = None
    start_time = time.perf_counter()

    with sf.SoundFile(AUDIO_FILE) as f:
        audio_data = f.read(dtype="int16")
    audio_bytes = audio_data.tobytes()

    try:
        async with websockets.connect(url, additional_headers=headers) as ws:
            
            async def sender():
                chunk_size = 4096
                for i in range(0, len(audio_bytes), chunk_size):
                    await ws.send(audio_bytes[i:i+chunk_size])
                    await asyncio.sleep(0.01)
                await ws.send(json.dumps({"type": "CloseStream"}))
            
            async def receiver():
                nonlocal first_interim_time, final_time
                async for msg in ws:
                    result = json.loads(msg)
                    if "channel" in result:
                        alt = result["channel"]["alternatives"][0]
                        transcript = alt["transcript"]
                        if transcript.strip():
                            now = time.perf_counter()
                            if first_interim_time is None:
                                first_interim_time = now
                                print(f"FIRST_INTERIM_MS={(now - start_time)*1000:.2f}")

                    if result.get("speech_final"):
                        final_time = time.perf_counter()
                        print(f"FINAL_MS={(final_time - start_time)*1000:.2f}")
                        break

            await asyncio.gather(sender(), receiver())
            
    except Exception as e:
        print(f"Error connecting to Deepgram: {e}")

asyncio.run(main())
