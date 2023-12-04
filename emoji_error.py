import emoji

userText = "This sentence contains emojis like 😂 and 😡."

userText_without_emojis = ''.join(c for c in userText if c not in emoji.UNICODE_EMOJI)

print(userText_without_emojis)