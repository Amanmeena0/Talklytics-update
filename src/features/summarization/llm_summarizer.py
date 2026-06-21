import os
from google import genai
from src.features.engagement.tracker import EngagementRecord

class LLMSummarizer:
    def __init__(self):
        self.api_key = os.environ.get("GEMINI_API_KEY")
        if self.api_key:
            self.client = genai.Client(api_key=self.api_key)
        else:
            self.client = None

    def generate_summary(self, records: list[EngagementRecord]) -> str:
        if not records:
            return "No conversation recorded."

        transcript_lines = []
        for r in records:
            if r.transcript.strip():
                time_str = r.timestamp.strftime("%H:%M:%S")
                transcript_lines.append(f"[{time_str}] Score: {r.score}/5 - {r.transcript}")
        
        full_transcript = "\n".join(transcript_lines)

        if not self.client:
            return (
                "⚠️ **GEMINI_API_KEY not found in environment.**\n\n"
                "To enable AI summaries, please set your Gemini API key as an environment variable.\n\n"
                "### Raw Transcript Captured:\n\n" + full_transcript
            )

        prompt = f"""
You are an expert Sales Coach and Analyst. 
Analyze the following transcript of a live sales/coaching call and generate a structured Post-Call Summary.

Please include:
1. **Executive Summary**: A brief 2-3 sentence overview of the call.
2. **BANT Analysis**: Budget, Authority, Need, and Timeline (if mentioned or implied).
3. **Key Objections**: Any hesitations or objections raised by the prospect.
4. **Next Steps**: Recommended follow-up actions and a short draft of a follow-up email.

Transcript:
{full_transcript}
"""
        try:
            response = self.client.models.generate_content(
                model='gemini-2.5-flash',
                contents=prompt,
            )
            return response.text
        except Exception as e:
            return f"❌ Failed to generate summary. Error: {str(e)}"
