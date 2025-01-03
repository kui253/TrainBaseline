DIR2NAME = {
    "mathqa": "../hf_datasets/allenai/math_qa/math_qa.py",
    "pubmedqa": "../hf_datasets/bigbio/pubmed_qa/pubmed_qa.py",
    "agieval": "../hf_datasets/agieval/agieval.py",
    "winogrande": "../hf_datasets/winogrande/winogrande.py",
    "arc_easy": "../hf_datasets/allenai/ai2_arc/ARC-Easy",
    "arc_challenge": "../hf_datasets/allenai/ai2_arc/ARC-Challenge",
}
SYS_PROMPT4DS = {
    "mathqa": "Solve the given math problem, and output the rationale and your choice in the format: Rationale: ... Answer: a/b/c/d/e. Do not output any other information",
    "pubmedqa": "Read the meterials, and answer the corresponding question. Output a yes/no with a brief description(reason)",
    "agieval": "Read the meterials, and answer the corresponding question. Output your choice in the format: Answer: A/B/C/D/E. Do not output any other information",
    "winogrande": "Choose the correct word to fill in the underlined parts (marked by _) of a given sentence, output your choice in the format: Answer: 1/2. Do not output any other information",
    "arc_easy": "None",
    "arc_challenge": "none",
}
