import os
import re
import tqdm
import pandas as pd
from ollama_setup import run_llm


class SpeakerRoleBaseline:
    def __init__(self):
        # Option to hash speaker names (set to True if needed)
        self.is_hash_speakers = False
        self.max_retries = 3

        # The static roles description remains unchanged.
        self.roles_description = """
You are an expert in analyzing conversations and assigning speaker roles.

The roles are:
- Protagonist: Leads the discussion and asserts authority, actively driving the conversation forward.
- Supporter: Offers encouragement, help, or positive reinforcement to other speakers.
- Neutral: Participates passively, often giving straightforward responses or reacting without adding substantial direction or conflict.
- Gatekeeper: Facilitates smooth communication, guides turn-taking, or helps clarify misunderstandings, ensuring the conversation remains balanced and productive.
- Attacker: Challenges others, expresses skepticism, or undermines the ideas and confidence of other speakers, introducing tension or conflict into the interaction.

When assigning roles, provide a brief but insightful justification that explains why the speaker's utterances, tone, or mannerisms align with the chosen role.
Reference their specific contributions in the conversation and connect them with the chosen role’s qualities. Avoid generic statements; 
give reasons rooted in the conversation’s content and character dynamics from the "Friends" universe.

For only one speaker, output only their role and a justification. Do not include any code, programming syntax, or commentary outside the specified format.
Do not repeat information about the roles beyond what is necessary for justification. Keep your response focused solely on the speaker names, their chosen roles, and the detailed justifications.

Format the output as a JSON file as follows:
"speaker_name":
"role": "chosen_role"
"justification": "detailed reason for choosing this role, referencing the utterances and character context"
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
        Builds the prompt using the new template, calls the LLM with retry logic, and returns parsed results.
        """
        prompt = self.generate_prompt(conversation, sr_no_list, dialogue_id, speakers_list)
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
                parsed_results = self.parse_response(response, sr_no_list, speakers_list, prompt, dialogue_id)
                # If all roles are successfully assigned (i.e. no "Error" returned), annotate and return.
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
        # If failed after all retries, return an error entry for each speaker in the dialogue.
        return [{
            "Sr No.": sr_no,
            "Speaker": speaker,
            "Dialogue_ID": dialogue_id,
            "Role": "Error",
            "Justification": f"Failed after {self.max_retries} attempts.",
            "Prompt": prompt,
            "Response": response
        } for sr_no, speaker in zip(sr_no_list, speakers_list)]

    def parse_response(self, response, sr_no_list, speakers_list, prompt, dialogue_id):
        """
        Parses the JSON response returned by the LLM and extracts the relevant fields.
        """
        speaker_match = re.search(r'"speaker_name":\s*"([^"]+)"', response)
        role_match = re.search(r'"role":\s*"([^"]+)"', response)
        justification_match = re.search(r'"justification":\s*"([^"]+)"', response)
        if self.is_hash_speakers:
            speakers_for_validation, _ = self._hash_speakers(speakers_list)
        else:
            speakers_for_validation = speakers_list

        # Fallback: if no speaker name is returned, use the first speaker from the conversation.
        speaker = speaker_match.group(1) if speaker_match else (
            speakers_for_validation[0] if speakers_for_validation else "Unknown")
        sr_no_str = ",".join(map(str, sr_no_list)) if sr_no_list else ""
        result = {
            "Sr No.": sr_no_str,
            "Speaker": speaker,
            "Dialogue_ID": dialogue_id,
            "Role": role_match.group(1) if role_match else None,
            "Justification": justification_match.group(1) if justification_match else None,
            "Prompt": prompt,
            "Response": response
        }
        # Return as a list to allow concatenation with other results.
        return [result]

    def _hash_speakers(self, speakers_list):
        """
        Dummy implementation of speaker hashing.
        Replace this with your actual hashing logic if required.
        """
        hashed = [f"Speaker_{i}" for i, _ in enumerate(speakers_list)]
        return hashed, None


# Example usage of the modified SpeakerRoleBaseline class within a process_data function:
def process_data(mode, model_instance: SpeakerRoleBaseline, output_file_suffix):
    """
    Processes input CSV data for SpeakerRoleBaseline.
    Groups the dialogues by Dialogue_ID, calls the model to assign roles,
    and appends (or writes) the results to a CSV file.
    """
    # Define your file paths (update these constants as needed)
    TRAIN_PATH = "path_to_train.csv"
    TEST_PATH = "path_to_test.csv"
    FINAL_SAVE_DIR = "final_save_directory"

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

    print(f"Results saved to {output_file}")
    return existing_df
