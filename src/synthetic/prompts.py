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