import torch
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset

class MMLUProDataset(Dataset):
    """
    A PyTorch Dataset wrapper for the MMLU-Pro dataset from Hugging Face.
    MMLU-Pro extends MMLU with more difficult questions and up to 10 choices.
    """
    def __init__(self, split="test", tokeniser=None):
        # Load dataset from Hugging Face
        # The MMLU-Pro dataset primarily uses the 'test' and 'validation' splits.
        self.dataset = load_dataset("TIGER-Lab/MMLU-Pro")[split]
        self.tokeniser = tokeniser
        
    def __len__(self):
        return len(self.dataset)

    def format_prompt(self, item):
        """
        Formats the question and its multiple choices into a prompt string.
        """
        question = item['question']
        options = item['options']
        
        # Format typical zero-shot prompt
        prompt = f"Question: {question}\nOptions:\n"
        letters = "ABCDEFGHIJ" # MMLU-Pro has up to 10 options
        for idx, option in enumerate(options):
            prompt += f"{letters[idx]}. {option}\n"
        prompt += "Answer: "
        return prompt
        
    def __getitem__(self, idx):
        item = self.dataset[idx]
        prompt = self.format_prompt(item)
        
        if self.tokeniser:
            # Tokenise the prompt
            inputs = self.tokeniser(prompt, return_tensors="pt", truncation=True, max_length=4096)
            return {
                "input_ids": inputs["input_ids"].squeeze(0),
                "attention_mask": inputs["attention_mask"].squeeze(0),
                "answer_index": item["answer_index"],
                "answer": item["answer"],
                "category": item["category"]
            }
        else:
            # Return raw string prompt if no tokeniser is provided
            return {
                "prompt": prompt,
                "answer_index": item["answer_index"],
                "answer": item["answer"],
                "category": item["category"]
            }

def custom_collate_fn(batch, tokeniser=None):
    """
    Custom collate function to handle variable length tokenised inputs or raw strings.
    """
    if "input_ids" in batch[0]:
        input_ids = [item["input_ids"] for item in batch]
        attention_mask = [item["attention_mask"] for item in batch]
        answer_index = [item["answer_index"] for item in batch]
        answers = [item["answer"] for item in batch]
        categories = [item["category"] for item in batch]
        
        if tokeniser and tokeniser.pad_token_id is not None:
            input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokeniser.pad_token_id)
            attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0)
        else:
            # If no pad_token_id is set, you might want to handle it or let it fail
            # depending on your specific requirements
            pass 
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "answer_index": torch.tensor(answer_index),
            "answer": answers,
            "category": categories
        }
    else:
        # Just return the lists if not tokenised
        prompts = [item["prompt"] for item in batch]
        answer_index = [item["answer_index"] for item in batch]
        answers = [item["answer"] for item in batch]
        categories = [item["category"] for item in batch]
        return {
            "prompt": prompts,
            "answer_index": torch.tensor(answer_index),
            "answer": answers,
            "category": categories
        }

def get_mmlu_pro_dataloader(split="test", batch_size=8, tokeniser=None, num_workers=4):
    """
    Creates and returns a PyTorch DataLoader for the MMLU-Pro dataset.
    """
    dataset = MMLUProDataset(split=split, tokeniser=tokeniser)
    
    # We use a lambda to pass the tokeniser into the collate_fn
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == "train"), # MMLU-Pro is mostly test/val
        num_workers=num_workers,
        collate_fn=lambda b: custom_collate_fn(b, tokeniser)
    )
    return dataloader

if __name__ == "__main__":
    # Simple test script
    # To run this you need to install `datasets`: pip install datasets
    print("Loading raw MMLU-Pro dataloader (no tokeniser)...")
    
    # MMLU-Pro typically has 'validation' and 'test' splits
    dataloader = get_mmlu_pro_dataloader(split="validation", batch_size=2, num_workers=0)
    
    for batch in dataloader:
        print("Prompt Sample:", batch["prompt"][0])
        print("Expected Answer:", batch["answer"][0])
        print("Answer Index:", batch["answer_index"][0])
        print("Category:", batch["category"][0])
        break
