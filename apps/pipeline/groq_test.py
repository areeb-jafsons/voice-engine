import os
import time
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])

for i in range(3):
    start = time.perf_counter()
    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": "Say hello briefly."}],
        stream=True,
    )

    first_token_time = None

    for chunk in stream:
        if first_token_time is None:
            first_token_time = time.perf_counter()
            break

    ttft = (first_token_time - start) * 1000
    print(f"Run {i+1} TTFT: {ttft:.2f} ms")
