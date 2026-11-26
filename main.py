import dotenv
import os
from openai import OpenAI

dotenv.load_dotenv()

class Analyzer:
    model: str = "openai/gpt-oss-20b"
    api_key: str | None = None
    base_url: str = "https://api.groq.com/openai/v1"
    def __init__(self):
        key = self.api_key or os.environ.get("GROQ_API_KEY")
        if not key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Add it to your environment or .env file."
            )
        self._llm = OpenAI(api_key=key, base_url=self.base_url)

        self.messages = [
            {
                'role': 'system',
                'content': 'Classify the sentiment of the text as positive, negative, or neutral'
            },
            {
                'role': 'user',
                'content': 'I absolutely loved this restaurant!'
            },
            {
                'role': 'assistant',
                'content': 'sentiment: positive'
            },
            {
                'role': 'user',
                'content': 'the wait time was way too long'
            },
            {
                'role': 'assistant',
                'content': 'sentiment: negative'
            },
            {
                'role': 'user',
                'content': 'it was an average experience, nothing special'
            },
            {
                'role': 'assistant',
                'content': 'sentiment: neutral'
            }
        ]

    def sentiment_analysis(self, text):
        response = self._llm.chat.completions.create(
            model='openai/gpt-oss-20b',
            temperature=0.1,
            messages=[
                *self.messages,
                {'role': 'user', 'content': text}
            ]
        )
        return response.choices[0].message.content

def main():
    analyzer = Analyzer()
    while True:
        user_input = input("Say something and I will analyze how do you feel today (type Q to quit): ").strip()
        if user_input.lower() == "q":
            print("Goodbye!")
            break
        if not user_input:
            continue

        try:
            sentiment = analyzer.sentiment_analysis(user_input)
        except RuntimeError as exc:
            print(f"[error] {exc}")
            continue

        print(sentiment)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    main()

