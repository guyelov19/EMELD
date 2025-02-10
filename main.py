from approach2 import Approach2, process_data as process_data_approach2
from approach3 import Approach3, process_data as process_data_approach3
from utils import load_dataset

def main():
    print("Loading dataset...")
    train_df, test_df = load_dataset()
    print(test_df)

    mode = "test"  # Change to "train" if needed
    max_retries = 20

    # --- Run Approach 2 ---
    # approach2_instance = Approach2(max_retries, is_hash_speakers=True)
    # print("Processing data using Approach 2 ...")
    # process_data_approach2(mode, approach2_instance, output_file_suffix="approach2")
    # print("Approach 2 processing complete.\n")

    # --- Run Approach 3 ---
    # approach3_instance = Approach3(max_retries, is_hash_speakers=True)
    # print("Processing data using Approach 3 ...")
    # process_data_approach3(mode, approach3_instance, output_file_suffix="approach3", group_by=["Episode", "Season"], is_hash_speakers=approach3_instance.is_hash_speakers)
    # print("Approach 3 processing complete.")

if __name__ == "__main__":
    main()
