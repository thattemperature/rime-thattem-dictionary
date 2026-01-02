import os
from openai import OpenAI
from typing import List, Dict


def devide(word_list: List[str]) -> Dict[str, str]:

    client = OpenAI(
        api_key=os.environ.get("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {
                "role": "system",
                "content": "You are a language expert who divides English words into syllables.\n"
                + "Given an English word, return the word's syllables in the correct order, separated by spaces.\n"
                + "The input is a word list, each line is a word. Your reply should has exact same line number.\n"
                + "Note: the only thing you can do is to insert spaces.\n"
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
                "content": "\n".join(word_list),
            },
        ],
        stream=False,
    )

    result = response.choices[0].message.content
    result = result.split("\n")

    if len(result) != len(word_list):
        raise Exception(
            f"Input length: {len(word_list)}, output length: {len(result)}"
        )
    for index, word in enumerate(word_list):
        if not all((c.isalpha() or c in "-_' " for c in word)):
            word_list[index] = "!"

    return dict(zip(word_list, result))
