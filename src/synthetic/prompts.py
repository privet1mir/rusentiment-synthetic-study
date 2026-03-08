RAW_SENTIMENT_PROMPT = """
Generate a short Russian social media post (similar to a VKontakte message or comment)
for a sentiment classification task.

Avoid repeating common templates and generate diverse topics.

Target label: {label}

Labels:
negative — negative emotion, complaint, frustration, sadness
neutral — factual statement, observation, or emotionally neutral message
positive — happiness, excitement, affection, praise

Requirements:
- The text should look like a real social media post or comment.
- Use informal language typical for Russian social networks.
- Emojis are allowed but should appear only sometimes
- Slang, repeated punctuation (!!!, ...), lowercase style, and casual tone are allowed.
- Length: 5–30 words.
- Vary topics (friends, music, work, daily life, internet, relationships, random thoughts).

Return strictly in JSON format:

{{
  "text": "<russian social media post>",
  "label": "{label}"
}}
"""


FEW_SHOT_PROMPTS_MAPPER = {
    "positive": [
        "Ураааа!! Мне гипс сняли!!! Скоро снова всех побеждать начну))",
        "Чай и любимые зефирки! Счастье в мире есть ^_^",
        "Чудесный денёк сегодня!!! не так ли?:-)",
        "Я рада, что у меня есть человек, который всегда меня поймет. Человек, с которым совершенно непонятным образом у меня всегда сходятся мысли...)))"
    ],
    "negative": [
        "Чтоб ты слюнями захлебнулся, пес смердячий!.",
        "НЕНАВИЖУ СВОЮ РАБОТУ ! ! !",
        "Какой кошмар!!!! Не хочу такое лето!!!!😿",
        "Печально когда проявляешь чувства, строишь планы, надеешься на что то, а тут бац и все зря!((("
    ],
    "neutral": [
        "завтра первый экзамен!!!",
        "Подскажите где на ней покататься то можно?",
        "Опять потерял ключи от машины, постоянно что то теряю, интересно это нормально, или пора к врачу?",
        "В Армении пасмурно, похоже весь день так будет. Как у вас погода? :)"
    ]
}

FEW_SHOT_SENTIMENT_PROMPT = """
Generate a short Russian social media post (similar to a VKontakte message or comment)
for a sentiment classification task.

Target label: {label}

Labels:
negative — negative emotion, complaint, frustration, sadness
neutral — factual statement, observation, or emotionally neutral message
positive — happiness, excitement, affection, praise

Examples (label = {label}):

{examples}

Requirements:
- The text should look like a real Russian social media post or comment.
- Use informal language typical for Russian social networks.
- Emojis may appear occasionally but should not appear in every example.
- Slang, repeated punctuation (!!!, ...), lowercase style, and casual tone are allowed.
- Length: 5–30 words.
- Vary topics (friends, work, music, daily life, internet, relationships, random thoughts).
- Do not repeat or closely paraphrase the examples above.

Return strictly in JSON format:

{{
  "text": "<russian social media post>",
  "label": "{label}"
}}
"""

TAXONOMY_SENTIMENT_PROMPT = """
Generate a short Russian social media post (similar to a VKontakte message or comment)
for a sentiment classification task.

Target label: {label}
Topic: {topic}

Labels:
negative — negative emotion, complaint, frustration, sadness
neutral — factual statement, observation, or emotionally neutral message
positive — happiness, excitement, affection, praise

Topic description:
The post should clearly relate to the topic "{topic}".

Requirements:
- The text should look like a real Russian social media post or comment.
- Use informal language typical for Russian social networks.
- Emojis may appear occasionally but should not appear in every example.
- Slang, repeated punctuation (!!!, ...), lowercase style, and casual tone are allowed.
- Length: 5–30 words.
- The content should naturally relate to the topic "{topic}".
- Avoid repeating templates.

Return strictly in JSON format:

{{
  "text": "<russian social media post>",
  "label": "{label}"
}}
"""
