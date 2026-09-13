import time
from groq import RateLimitError
from app.groq_client import client


class LLM:

    def complete(self, prompt: str):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                start = time.perf_counter()
                response = client.chat.completions.create(
                    model="openai/gpt-oss-20b",
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                latency = time.perf_counter() - start
                return response.choices[0].message.content, latency

            except RateLimitError:
                wait = 2 ** attempt
                print(f"Rate limit. Retrying in {wait}s..")
                time.sleep(wait)

        # Only reached after every attempt failed (was inside the loop before,
        # which raised after the first rate limit instead of retrying)
        raise Exception("Maximum retries exceeded.")


    def stream(self, prompt: str):
        stream = client.chat.completions.create(
            model="openai/gpt-oss-20b",
            messages=[
                {"role": "user", "content": prompt}
            ],
            stream=True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
