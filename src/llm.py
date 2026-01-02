import os
from openai import OpenAI


def devide(word: str) -> str:

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a language expert who divides English words into syllables. Given an English word, return the word's syllables in the correct order, separated by spaces."
                + "Note: the only thing you can do is insert spaces."
                + """
### Examples:
Input:
pronunciation
Output:
pro nun ci a tion

Input:
don't
Output:
do n't

Input:
aren't
Output:
are n't

Input:
alpha-based
Output:
al pha - based

Input:
large_language_model
Output:
large _ lan guage _ mod le
""",
            },
            {
                "role": "user",
                "content": word,
            },
        ],
        stream=False,
    )

    if not all(c.isalpha() or c in "-_' " for c in word):
        return "!"

    return response.choices[0].message.content
