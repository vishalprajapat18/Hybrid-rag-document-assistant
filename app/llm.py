import json
import time
from groq import RateLimitError
from app.groq_client import client
from sentence_transformers import SentenceTransformer
from app.models import ChatRequest, ChatResponse
embedding_model = SentenceTransformer("BAAI/bge-small-en-v1.5")


class LLM:

    def complete(self, prompt: str) -> str:
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
              end = time.perf_counter()
              latency = end-start
              return response.choices[0].message.content ,latency
          
          except RateLimitError:
              wait = 2**attempt
              print(f"Rate Limit. retrying in{wait}s..")
              time.sleep(wait)

          raise Exception("Maximum retries exceeded.")
  


    def stream(self,prompt:str):
        stream = client.chat.completions.create (
            model ="openai/gpt-oss-20b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream = True
        )
        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
              yield chunk.choices[0].delta.content


    #def extract(self,text:str) -> CustomerFeedback:
#
 #       prompt= f"""
  #     Extract the following customer feedback.
#
 #        Return only valid JSON with:
  #       sentiment 
   #      category
    #     urgency

     #    Feedback:
      #   {text}"""

       ##model="openai/gpt-oss-20b",
        #messages=[
         #   {
          #      "role": "user",
           #     "content": prompt
            #}
     #   ]
    #)
      #  data = json.loads(response.choices[0].message.content)

       # return CustomerFeedback(**data)
        

    def embed(self, text: str) -> list[float]:

      vector = embedding_model.encode(text)

      return vector.tolist()  