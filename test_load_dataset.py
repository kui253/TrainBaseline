from datasets import load_dataset

DIR2NAME = {
    "specialty": {
        "mathqa": "../hf_datasets/allenai/math_qa/math_qa.py",
        "pubmedqa": "../hf_datasets/bigbio/pubmed_qa/pubmed_qa.py",
        "agieval": "../hf_datasets/agieval/agieval.py",
        "winogrande": "../hf_datasets/winogrande/winogrande.py",
    },
    "generality": {
        "arc_easy": "../hf_datasets/allenai/ai2_arc/ARC-Easy",
        "arc_challenge": "../hf_datasets/allenai/ai2_arc/ARC-Challenge",
    },
}

datas = load_dataset(DIR2NAME["specialty"]["pubmedqa"])
import pdb

pdb.set_trace()
