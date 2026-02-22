import asyncio
import os
from groq import Groq

# Use the API key directly for now, or from environment variable if set
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
client = Groq(api_key=GROQ_API_KEY)

async def keep_groq_warm():
    """Background task to keep the Groq LLM model warm."""
    print("Starting Groq keep-warm background task...")
    while True:
        try:
            # We use the sync client in an async thread implicitly or explicitly, 
            # but since network I/O might block, in a production app with async endpoints,
            # using AsyncGroq is preferred. For this keep-warm loop, this is sufficient.
            client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                stream=False,
            )
            print("Groq model pinged to keep warm.")
        except Exception as e:
            print(f"Error pinging Groq model: {e}")

        await asyncio.sleep(120)  # every 2 minutes

async def main():
    print("LiveKit test placeholder")
    
    # Start the background keep-warm task
    warmup_task = asyncio.create_task(keep_groq_warm())
    
    # Run indefinitely (or until cancelled) to simulate a running pipeline
    try:
        # Keep the main loop alive so the background task runs
        while True:
            await asyncio.sleep(3600)
    except asyncio.CancelledError:
        print("Pipeline shutting down...")
    finally:
        warmup_task.cancel()

if __name__ == "__main__":
    import asyncio
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nShutdown requested")
