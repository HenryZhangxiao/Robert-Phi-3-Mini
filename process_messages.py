import json

INPUT_FILE = ""   # raw chat export
OUTPUT_FILE = ""    # dataset for training

YOUR_NAME = ""              # your Discord username
FRIEND_NAME = ""               # your friend's Discord username

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    data = json.load(f)["messages"]

id_map = {msg["id"]: msg for msg in data}
dataset = []

# --- For server processing ---
for msg in data:
    # Only look at friend's replies
    if msg["author"]["name"] == FRIEND_NAME and msg["type"] == "Reply":
        ref_id = msg["reference"]["messageId"]
        if ref_id and ref_id in id_map:
            original = id_map[ref_id]

            prompt = f"User ({original['author']['name']}): {original['content']}\nFriend:"
            response = msg["content"]

            if response.strip():
                dataset.append({"prompt": prompt, "response": response})

# --- For DM processing ---
# for i in range(len(data) - 1):
#     curr_msg = data[i]
#     next_msg = data[i + 1]

#     # only take (you -> friend) pairs
#     if curr_msg["author"]["name"] == YOUR_NAME and next_msg["author"]["name"] == FRIEND_NAME:
#         prompt = f"User: {curr_msg['content']}\nFriend:"
#         response = next_msg["content"]

#         dataset.append({"prompt": prompt, "response": response})

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print(f"Saved {len(dataset)} prompt-response pairs to {OUTPUT_FILE}")
