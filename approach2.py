import os
import re
import tqdm
import pandas as pd
from ollama_setup import run_llm
from base_role_approach import BaseRoleApproach
from config import TRAIN_PATH, TEST_PATH, FINAL_SAVE_DIR

class Approach2(BaseRoleApproach):
    """
    Inherits the common functionality from BaseRoleApproach.
    """
    def generate_prompt(self, conversation, sr_no_list, dialogue_id, speakers_list):
        """
        Generates the prompt text for the LLM.
        If self.is_hash_speakers is True, speaker names are replaced with hashed identifiers.
        The conversation is a list of (Speaker, Utterance) tuples.
        """
        if self.is_hash_speakers:
            hashed_speakers_list, _ = self._hash_speakers(speakers_list)
            conversation_text = "\n".join(
                f"Sr No. {sr_no}, {hashed_speaker}: \"{utterance}\""
                for sr_no, (_, utterance), hashed_speaker in zip(sr_no_list, conversation, hashed_speakers_list)
            )
        else:
            conversation_text = "\n".join(
                f"Sr No. {sr_no}, {speaker}: \"{utterance}\""
                for sr_no, (speaker, utterance) in zip(sr_no_list, conversation)
            )
        prompt = (
            f"{self.roles_description}\n\n"
            f"Here is the context of the entire dialogue with Dialogue_ID {dialogue_id}:\n"
            f"{conversation_text}\n\n"
            "Identify the role and provide justifications for each speaker. Ensure each response includes "
            "'Sr No.', 'Speaker', 'Role', and 'Justification'. Additionally, include the length of each "
            "dialogue utterance in the response."
        )
        return prompt

    def assign_roles(self, conversation, sr_no_list, speakers_list, dialogue_id):
        """
        Builds the prompt, calls the LLM with retry logic, and returns parsed results.
        """
        prompt = self.generate_prompt(conversation, sr_no_list, dialogue_id, speakers_list)
        if self.is_hash_speakers:
            hashed_speakers_list, _ = self._hash_speakers(speakers_list)
            speakers_for_validation = hashed_speakers_list
        else:
            speakers_for_validation = speakers_list

        template = (
            "You are an assistant specialized in analyzing dialogue and identifying speaker roles. "
            "Answer the following question in a valid JSON format.\n\n"
            "Question: {question}\n\n"
            "Answer: Think step by step and provide a JSON object following the specified format."
        )

        attempts = 0
        response = None
        while attempts < self.max_retries:
            try:
                response = run_llm(prompt, template)
                parsed_results = self.parse_response(response, sr_no_list, speakers_for_validation, prompt, dialogue_id)
                if all(result["Role"] != "Error" for result in parsed_results):
                    # Annotate valid results.
                    for result in parsed_results:
                        result["Dialogue_ID"] = dialogue_id
                        result["Prompt"] = prompt
                        result["Response"] = response
                    return parsed_results
            except Exception as e:
                if "Connection refused" in str(e):
                    raise RuntimeError(f"Critical Error: {str(e)}") from e
            attempts += 1

        print(f"Failed to assign roles for Dialogue_ID {dialogue_id} after {self.max_retries} attempts.", flush=True)
        return [{
            "Sr No.": sr_no,
            "Speaker": speaker,
            "Dialogue_ID": dialogue_id,
            "Role": "Error",
            "Justification": f"Failed after {self.max_retries} attempts.",
            "Prompt": prompt
        } for sr_no, speaker in zip(sr_no_list, speakers_for_validation)]

def process_data(mode, model_instance: Approach2, output_file_suffix):
    """
    Processes input CSV data for Approach 2.
    Groups the dialogues by Dialogue_ID, calls the model to assign roles,
    and appends (or writes) the results to a CSV file.
    """
    input_path = TRAIN_PATH if mode == 'train' else TEST_PATH
    input_df = pd.read_csv(input_path)
    # Keep only the required columns.
    input_df = input_df[['Sr No.', 'Dialogue_ID', 'Speaker', 'Utterance']]
    grouped_conversations = input_df.groupby('Dialogue_ID')

    output_file = os.path.join(
        FINAL_SAVE_DIR,
        f"{mode}_{output_file_suffix}{'_hashed' if model_instance.is_hash_speakers else ''}.csv"
    )

    # Create the output file with header if it does not exist.
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Sr No.,Speaker,Dialogue_ID,Role,Justification,Prompt,Response\n")

    existing_df = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()

    for dialogue_id, group in tqdm.tqdm(grouped_conversations, desc="Processing dialogues"):
        # Skip dialogues already processed successfully.
        if dialogue_id in existing_df.get("Dialogue_ID", []):
            dialogue_entries = existing_df[existing_df["Dialogue_ID"] == dialogue_id]
            if "Error" not in dialogue_entries["Role"].values:
                continue
            existing_df = existing_df[existing_df["Dialogue_ID"] != dialogue_id]

        conversation_data = list(zip(group['Speaker'], group['Utterance']))
        sr_no_list = group['Sr No.'].tolist()
        speakers_list = group['Speaker'].tolist()

        roles = model_instance.assign_roles(conversation_data, sr_no_list, speakers_list, dialogue_id)
        res_df = pd.DataFrame(roles)
        existing_df = pd.concat([existing_df, res_df]).sort_values(by="Sr No.").reset_index(drop=True)
        existing_df.to_csv(output_file, index=False, encoding='utf-8')
