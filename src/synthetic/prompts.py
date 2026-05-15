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

DIVERSE_SENTIMENT_PROMPT = """
Generate a Russian social media comment (VK style).

Target label: {label}

Write like a real person, not like an AI.

---

The comment MUST be varied and NOT repetitive.

Avoid typical шаблоны:
- "ну да", "ну норм", "ну и", "да норм", "да обычный"
- generic safe phrases

---

The comment can be:
- very short (1–3 words) OR long and messy
- emotional OR neutral OR strange
- logical OR chaotic OR random

---

Style:
- informal, sloppy, imperfect
- slang, profanity allowed
- typos allowed
- repetitions (бляяя, дааа)
- broken punctuation (!!!???...)
- unfinished thoughts are OK
- weird or exaggerated tone is OK

---

Important:
The text should feel spontaneous, like written quickly without thinking.

Do NOT:
- explain anything
- write clean or well-structured sentences
- produce safe generic text

---

Label meaning:
negative — anger, hate, frustration, sadness
neutral — everyday thoughts, weak emotion
positive — joy, affection, excitement

---

Length: 1–50 words

---

Return strictly JSON:

{{
  "text": "<russian comment>",
  "label": "{label}"
}}
"""

LATENT_CONFIG_PROMPT = """
Generate a realistic Russian VK-style social media post.

Target sentiment label: {label}

Latent generation configuration:

Topic: {topic}
Intent: {intent}
Emotion: {emotion}
Style: {style}
Structure: {structure}
VK archetype: {archetype}

Surface realism:
- capslock: {capslock}
- emoji usage: {emoji_usage}
- typos: {typos}
- profanity: {profanity}
- punctuation: {punctuation}

Requirements:
- Write like a real person from old VK/social media.
- The text should feel spontaneous and authentic.
- Avoid generic AI phrasing.
- The text may be messy, emotional, weird or fragmented.
- Do not explain the configuration.
- Do not make the text overly coherent.
- Avoid repetition of common templates.

Length:
1–50 words.

Return strictly JSON:

{{
  "text": "<russian vk-style text>",
  "label": "{label}"
}}
"""
