from openai import OpenAI
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

class GPTQueryEngine:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
    
    def query(self, query: str, context: str) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant specializing in tourism industry analysis. Provide clear, accurate answers based on the provided context."},
                    {"role": "user", "content": f"Context: {context}\n\nQuestion: {query}"}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Error querying GPT-4: {str(e)}")
            return None
