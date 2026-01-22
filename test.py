import pickle

with open('datasets/lmsys_chat1m_prompts_100k_cleaned.pkl', 'rb') as f:
    data = pickle.load(f)

print(len(data))