import os
import re
import tqdm
import pandas as pd
from ollama_setup import run_llm
from base_role_approach import BaseRoleApproach
from config import TRAIN_PATH, TEST_PATH, FINAL_SAVE_DIR

class Approach3(BaseRoleApproach):
    """
    Extends BaseRoleApproach and adds support for including conversation durations and connection summaries.
    """
    def generate_prompt(self, conversation, sr_no_list, duration_list, dialogue_id, speakers_list, connection_summary=None):
        """
        Generates the dialogue prompt.
        Optionally includes a connection summary.
        """
        if self.is_hash_speakers:
            _, hashed_speakers_list = self._hash_speakers(speakers_list)
        else:
            hashed_speakers_list = speakers_list

        conversation_text = "\n".join(
            f"Sr No. {sr_no}, {hashed_speaker} ({round(duration, 2)}s): \"{utterance}\""
            for sr_no, duration, (speaker, utterance), hashed_speaker
            in zip(sr_no_list, duration_list, conversation, hashed_speakers_list)
        )

        connection_text = ""
        if connection_summary:
            connection_text = "\n\nHere is a detailed summary of how the participants interact with specific individuals:\n"
            connection_text += "\n".join(
                f"- Speaker `{conn['Speaker_Response']}` when interacting with `{conn['Speaker_Responded_To']}`:\n"
                f"  • Avg Duration of Response: {conn['Response_Duration']:.2f}s\n"
                f"  • Avg Words Used: {conn['Words_in_Response']:.2f}\n"
                f"  • Avg Letters Used: {conn['Letters_in_Response']:.2f}\n"
                f"  • Sentiments Expressed: {', '.join([f'{sent[0]} ({sent[1]}%)' for sent in conn['Sentiment_in_Response']])}\n"
                f"  • Emotions Displayed: {', '.join([f'{emo[0]} ({emo[1]}%)' for emo in conn['Emotion_in_Response']])}"
                for conn in connection_summary.get('Connection_Summary', [])
            )
            connection_text += "\n\nNow, let's look at an overall view of how each speaker communicates in general, across all their conversations:\n"
            connection_text += "\n".join(
                f"- Speaker `{part['Speaker_Response']}` (overall communication):\n"
                f"  • Avg Duration of Response: {part['Response_Duration']:.2f}s\n"
                f"  • Avg Words Used: {part['Words_in_Response']:.2f}\n"
                f"  • Avg Letters Used: {part['Letters_in_Response']:.2f}\n"
                f"  • Sentiments Expressed: {', '.join([f'{sent[0]} ({sent[1]}%)' for sent in part['Sentiment_in_Response']])}\n"
                f"  • Emotions Displayed: {', '.join([f'{emo[0]} ({emo[1]}%)' for emo in part['Emotion_in_Response']])}"
                for part in connection_summary.get('Participants_Summary', [])
            )
            connection_text += (
                "\n\nKey Insight:\n"
                "The `Connection Summary` provides a focused view of how speakers communicate with specific individuals. "
                "In contrast, the `Participants Summary` reveals their general communication patterns across all interactions.\n"
                "Comparing these summaries can highlight whether speakers adjust their communication style based on the person they are speaking to."
            )

        prompt = (
            f"{self.roles_description}\n\n"
            f"Here is the context of the entire dialogue with Dialogue_ID {dialogue_id}:\n{conversation_text}\n"
            f"{connection_text}\n\n"
            "Identify the role and provide justifications for each speaker. Ensure each response includes "
            "'Sr No.', 'Speaker', 'Role', and 'Justification'. Additionally, include the length of each dialogue utterance in the response."
        )
        return prompt

    def assign_roles(self, conversation, sr_no_list, duration_list, speakers_list, dialogue_id, connection_summary=None):
        """
        Builds the prompt (including connection summaries if provided), calls the LLM, and returns parsed results.
        """
        prompt = self.generate_prompt(conversation, sr_no_list, duration_list, dialogue_id, speakers_list, connection_summary)
        if self.is_hash_speakers:
            _, speakers_for_validation = self._hash_speakers(speakers_list)
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
            "Prompt": prompt,
            "Response": response
        } for sr_no, speaker in zip(sr_no_list, speakers_for_validation)]

def process_data(mode, model_instance: Approach3, output_file_suffix, group_by, is_hash_speakers):
    """
    Processes the input CSV data for Approach 3.
    Loads the data (including time fields to compute duration), groups by Dialogue_ID,
    and calls the model to assign roles (with optional connection summaries).
    
    Note: For simplicity, this runner does not compute a full connection summary.
          You can extend this function to call your own summarization routines.
    """
    input_path = TRAIN_PATH if mode == 'train' else TEST_PATH
    input_df = pd.read_csv(input_path)
    # Convert StartTime and EndTime to datetime and compute Duration.
    input_df['StartTime'] = pd.to_datetime(input_df['StartTime'].str.replace(',', '.'))
    input_df['EndTime'] = pd.to_datetime(input_df['EndTime'].str.replace(',', '.'))
    input_df['Duration'] = (input_df['EndTime'] - input_df['StartTime']).dt.total_seconds()
    # Keep only needed columns.
    input_df = input_df[['Sr No.', 'Dialogue_ID', 'Speaker', 'Utterance', 'Duration']]
    grouped_conversations = input_df.groupby('Dialogue_ID')

    output_file = os.path.join(
        FINAL_SAVE_DIR,
        f"{mode}_{output_file_suffix}{'_hashed' if model_instance.is_hash_speakers else ''}.csv"
    )
    if not os.path.exists(output_file):
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("Sr No.,Speaker,Dialogue_ID,Role,Justification,Prompt,Response\n")
    existing_df = pd.read_csv(output_file) if os.path.exists(output_file) else pd.DataFrame()

    for dialogue_id, group in tqdm.tqdm(grouped_conversations, desc="Processing Dialogues"):
        if dialogue_id in existing_df.get("Dialogue_ID", []):
            dialogue_entries = existing_df[existing_df["Dialogue_ID"] == dialogue_id]
            if "Error" not in dialogue_entries["Role"].values:
                continue
            existing_df = existing_df[existing_df["Dialogue_ID"] != dialogue_id]

        conversation = list(zip(group['Speaker'], group['Utterance']))
        sr_no_list = group['Sr No.'].tolist()
        duration_list = group['Duration'].tolist()
        speakers_list = group['Speaker'].tolist()

        # For now, we set connection_summary to None.
        connection_summary = None  
        roles = model_instance.assign_roles(conversation, sr_no_list, duration_list, speakers_list, dialogue_id, connection_summary)
        res_df = pd.DataFrame(roles)
        existing_df = pd.concat([existing_df, res_df]).sort_values(by="Sr No.").reset_index(drop=True)
        existing_df.to_csv(output_file, index=False, encoding='utf-8')
