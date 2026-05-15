GENERATION_CONFIG = {
    # ---------------------------------------------------
    # HIGH-LEVEL SEMANTIC AXES
    # ---------------------------------------------------

    "topic": [
        "short_reaction",
        "daily_life",
        "relationships",
        "humor",
        "discussion",
        "creative",
        "toxicity",
        "storytelling",
        "philosophical",
        "question",
        "music",
        "other"
    ],

    "intent": [
        "reaction",
        "storytelling",
        "reflection",
        "complaint",
        "discussion",
        "confession",
        "joke",
        "rant",
        "observation",
        "asking"
    ],

    "emotion": [
        "positive",
        "neutral",
        "negative",
        "mixed",
        "nostalgic",
        "melancholic",
        "aggressive",
        "horny",
        "hopeful",
        "dramatic"
    ],

    "style": [
        "old_vk_wall_post",
        "messy_emotional",
        "capslock_chaos",
        "internet_slang",
        "poetic",
        "dramatic_prose",
        "normal_conversational",
        "drunk_post",
        "shitpost",
        "repost_commentary",
        "late_night_post",
        "music_status_core"
    ],

    "structure": [
        "very_short",
        "single_sentence",
        "multi_sentence",
        "long_paragraph",
        "poem_like",
        "dialogue_like",
        "fragmented_stream"
    ],

    # ---------------------------------------------------
    # CONDITIONAL SAMPLING
    # ---------------------------------------------------

    "topic_to_intents": {
        "short_reaction": [
            "reaction",
            "joke"
        ],

        "daily_life": [
            "observation",
            "storytelling",
            "complaint"
        ],

        "relationships": [
            "confession",
            "reflection",
            "storytelling"
        ],

        "humor": [
            "joke",
            "reaction",
            "shitpost"
        ],

        "discussion": [
            "discussion",
            "reflection",
            "rant"
        ],

        "creative": [
            "reflection",
            "confession",
            "storytelling"
        ],

        "toxicity": [
            "rant",
            "reaction",
            "complaint"
        ],

        "storytelling": [
            "storytelling",
            "observation"
        ],

        "philosophical": [
            "reflection",
            "discussion"
        ],

        "question": [
            "asking",
            "discussion"
        ],

        "other": [
            "observation",
            "reaction",
            "discussion"
        ]
    },

    "topic_to_emotions": {
        "relationships": [
            "positive",
            "nostalgic",
            "melancholic",
            "dramatic"
        ],

        "toxicity": [
            "negative",
            "aggressive"
        ],

        "humor": [
            "neutral",
            "positive",
            "mixed"
        ],

        "creative": [
            "melancholic",
            "nostalgic",
            "dramatic"
        ],

        "daily_life": [
            "neutral",
            "mixed",
            "positive"
        ],

        "short_reaction": [
            "positive",
            "neutral",
            "aggressive"
        ]
    },

    "topic_to_styles": {
        "short_reaction": [
            "normal_conversational",
            "internet_slang",
            "messy_emotional",
            "shitpost"
        ],
        "relationships": [
            "normal_conversational",
            "messy_emotional",
            "late_night_post",
            "internet_slang"
        ],
        "creative": [
            "poetic",
            "normal_conversational",
            "late_night_post"
        ],
        "humor": [
            "shitpost",
            "internet_slang",
            "repost_commentary"
        ],

        "discussion": [
            "normal_conversational",
            "old_vk_wall_post"
        ],

        "toxicity": [
            "normal_conversational",
            "messy_emotional",
            "internet_slang",
            "capslock_chaos"
        ]
    },

    "topic_to_structure": {
        "short_reaction": [
            "very_short",
            "single_sentence"
        ],

        "creative": [
            "poem_like",
            "long_paragraph"
        ],

        "storytelling": [
            "multi_sentence",
            "long_paragraph"
        ],

        "discussion": [
            "multi_sentence"
        ],

        "relationships": [
            "multi_sentence",
            "fragmented_stream"
        ]
    },

    # ---------------------------------------------------
    # NOISE / REALISM FEATURES
    # ---------------------------------------------------

    "surface_features": {
        "capslock": [
            "none",
            "light",
            "heavy"
        ],

        "emoji_usage": [
            "none",
            "light",
            "heavy"
        ],

        "typos": [
            "clean",
            "light_typos",
            "heavy_typos"
        ],

        "profanity": [
            "none",
            "light",
            "heavy"
        ],

        "punctuation": [
            "normal",
            "repeated_exclamations",
            "ellipsis_heavy",
            "chaotic"
        ],

        "internet_markers": [
            ":)",
            ":D",
            "xD",
            ")))",
            "ахах",
            "бля",
            "еее",
            "ору",
            "лол"
        ]
    },

    # ---------------------------------------------------
    # VK-SPECIFIC LATENT VIBES
    # ---------------------------------------------------

    "vk_archetypes": [
        "sad_night_post",
        "music_obsession_post",
        "relationship_wall_post",
        "drunk_existential_post",
        "aggressive_comment_war",
        "repost_with_commentary",
        "concert_hype",
        "random_life_fragment",
        "edgy_teen_post",
        "nostalgic_memory_post",
        "capslock_emotional_breakdown",
        "horny_flirty_post"
    ]
}