import os

# Define project directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FINAL_SAVE_DIR = os.path.join(RESULTS_DIR, "Approaches Annotations")
MANUAL_ANNOTATIONS_DIR = os.path.join(RESULTS_DIR, "Manual Annotations")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# File paths
TRAIN_PATH = os.path.join(DATA_DIR, "train_sent_emo.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_sent_emo.csv")
MANUAL_ANNOTATIONS_PATH = os.path.join(MANUAL_ANNOTATIONS_DIR, "annotated_test_sent_emo.csv")

# Ollama model name
OLLAMA_MODEL = "mistral"

# Hugging Face Token (ask user at runtime)
# HF_TOKEN = input("Enter your Hugging Face API token: ").strip()

VALID_ROLES = {'Protagonist', 'Supporter', 'Neutral', 'Gatekeeper', 'Attacker'}
