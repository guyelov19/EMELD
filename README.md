# NLP Project

This project implements multiple NLP approaches for analyzing conversations, speaker roles, and dialogue structure.

## **Project Structure**

```
NLP_Project/
│── main.py             # Entry point for running the project
│── approach1.py        # Placeholder for another NLP approach
│── approach2.py        # Speaker Role Assignment in conversations
│── approach3.py        # Summarizing Dialogue Connections
│── utils.py            # Utility functions
│── requirements.txt    # Required dependencies
│── README.md           # Project documentation
│── .gitignore          # Files to ignore in Git
└── data/               # Dataset storage
```

## **Setup Instructions**

### **1. Install Dependencies**

Make sure you have Python installed, then run:

```bash
pip install -r requirements.txt
```

### **2. Run the Project**

The project executes different NLP approaches. Run:

```bash
python main.py
```

## **Approaches**

### **1. Approach 1 (Placeholder)**

This file is a placeholder for another NLP approach.

### **2. Approach 2: Speaker Role Assignment**

This approach classifies speakers in a conversation into predefined roles:

- **Protagonist**: Leads the discussion, driving the conversation.
- **Supporter**: Encourages and reinforces other speakers.
- **Neutral**: Participates passively with minimal impact.
- **Gatekeeper**: Facilitates conversation flow and balance.
- **Attacker**: Challenges others, creating tension.

It uses an LLM (like **Mistral**) to analyze dialogue and assign roles.

To run only **Approach 2**:

```bash
python approach2.py
```

### **3. Approach 3: Dialogue Connection Summarization**

This method analyzes **who speaks to whom**, response durations, word usage, and sentiment. It provides:

- **Connection Summary**: How speakers interact with specific individuals.
- **Participants Summary**: Each speaker’s general communication style.

To run only **Approach 3**:

```bash
python approach3.py
```
