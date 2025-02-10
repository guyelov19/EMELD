import re
from config import VALID_ROLES

class BaseRoleApproach:
    """
    Shared functionality for approaches that assign speaker roles.
    Contains:
      - A common roles description.
      - Speaker hashing.
      - A generic response parser.
    """
    def __init__(self, max_retries=5, is_hash_speakers=False):
        self.max_retries = max_retries
        self.is_hash_speakers = is_hash_speakers
        self.roles_description = (
            "\nYou are an expert in analyzing conversations and assigning speaker roles. "
            "The following conversation is taken from various contexts, and your task is to assign roles "
            "to each speaker based on their utterances, including their role in the overall dialogue.\n\n"
            "The only roles possible are:\n"
            "- Protagonist: Leads the discussion and asserts authority, actively driving the conversation forward.\n"
            "- Supporter: Offers encouragement, help, or positive reinforcement to other speakers.\n"
            "- Neutral: Participates passively, often giving straightforward responses or reacting without adding substantial direction or conflict.\n"
            "- Gatekeeper: Facilitates smooth communication, guides turn-taking, or helps clarify misunderstandings, ensuring the conversation remains balanced and productive.\n"
            "- Attacker: Challenges others, expresses skepticism, or undermines the ideas and confidence of other speakers, introducing tension or conflict into the interaction.\n\n"
            "Format the output as a JSON file with the following structure, for each utterance in the dialogue:\n"
            "{\n"
            '\t"Sr No.": <Sr No.>,\n'
            '\t"Speaker": <Speaker_Name>,\n'
            '\t"Role": <Chosen_Role>,\n'
            '\t"Justification": <Detailed_Reason>\n'
            "}"
        )

    def _hash_speakers(self, speakers_list):
        """
        Returns a tuple (hashed_list, mapping) where each speaker is replaced
        by an identifier such as "Person A", "Person B", etc.
        """
        mapping = {}
        hashed_list = []
        next_char = ord('A')
        for speaker in speakers_list:
            if speaker not in mapping:
                mapping[speaker] = f"Person {chr(next_char)}"
                next_char += 1
            hashed_list.append(mapping[speaker])
        return hashed_list, mapping

    def parse_response(self, response, sr_no_list, speakers_list, prompt, dialogue_id):
        """
        Parses the JSON output from the LLM and validates that the order and values
        of speaker names and serial numbers match expectations.
        Returns a list of result dictionaries (or error dictionaries).
        """
        results = []
        try:
            matches = list(re.finditer(
                r'"Sr No\.":\s*(\d+),\s*"Speaker":\s*"([^"]+)",\s*"Role":\s*"([^"]+)",\s*"Justification":\s*"([^"]+)"',
                response
            ))
            if not matches:
                raise ValueError("Parsing failed: No valid matches found in the response.")

            model_sr_no_list = [int(match.group(1)) for match in matches]
            model_speakers_list = [match.group(2) for match in matches]
            roles_set = {match.group(3) for match in matches}

            if model_sr_no_list != sr_no_list:
                for sr_no, speaker in zip(sr_no_list, speakers_list):
                    results.append({
                        "Sr No.": sr_no,
                        "Speaker": speaker,
                        "Dialogue_ID": dialogue_id,
                        "Role": "Error",
                        "Justification": f"Sr No. mismatch: Expected {sr_no_list}, but got {model_sr_no_list}.",
                        "Prompt": prompt
                    })
                return results

            if model_speakers_list != speakers_list:
                for sr_no, speaker in zip(sr_no_list, speakers_list):
                    results.append({
                        "Sr No.": sr_no,
                        "Speaker": speaker,
                        "Dialogue_ID": dialogue_id,
                        "Role": "Error",
                        "Justification": f"Speaker mismatch: Expected {speakers_list}, but got {model_speakers_list}.",
                        "Prompt": prompt
                    })
                return results

            if not roles_set.issubset(VALID_ROLES):
                invalid_roles = roles_set - VALID_ROLES
                for sr_no, speaker in zip(sr_no_list, speakers_list):
                    results.append({
                        "Sr No.": sr_no,
                        "Speaker": speaker,
                        "Dialogue_ID": dialogue_id,
                        "Role": "Error",
                        "Justification": f"Invalid roles found: {invalid_roles}. Expected only {VALID_ROLES}.",
                        "Prompt": prompt
                    })
                return results

            # Build valid results.
            for match in matches:
                results.append({
                    "Sr No.": int(match.group(1)),
                    "Speaker": match.group(2),
                    "Role": match.group(3),
                    "Justification": match.group(4),
                    "Dialogue_ID": dialogue_id,
                    "Prompt": prompt
                })
        except Exception as e:
            results.append({
                "Sr No.": None,
                "Speaker": "Error",
                "Dialogue_ID": dialogue_id,
                "Role": "Error",
                "Justification": f"Error parsing response: {str(e)}",
                "Prompt": prompt
            })
        return results
