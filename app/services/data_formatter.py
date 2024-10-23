import pandas as pd
import re
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class DataFormatter:
    def format_response(self, answer: str) -> Dict[str, Any]:
        try:
            lines = answer.split('\n')
            
            if any(line.strip().startswith('•') for line in lines):
                data = []
                for line in lines:
                    if line.strip().startswith('•'):
                        parts = line.strip()[1:].split(':', 1)
                        if len(parts) == 2:
                            item = parts[0].strip()
                            values = parts[1].strip()
                            value_parts = re.findall(r'[\d,]+(?:\.\d+)?', values)
                            data.append([item] + value_parts)
                
                if data:
                    df = pd.DataFrame(data)
                    return {
                        'type': 'table',
                        'data': df.to_dict('records')
                    }
            
            return {
                'type': 'text',
                'data': answer
            }
        except Exception as e:
            logger.error(f"Error formatting response: {str(e)}")
            return {
                'type': 'text',
                'data': answer
            }