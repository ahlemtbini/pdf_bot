import json
import torch
from torch.utils.data import Dataset

class QADataset(Dataset):
    def __init__(self, json_file, tokenizer):
        with open(json_file, "r") as file:
            self.data = json.load(file)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        encoding = self.tokenizer(
            item['question'],
            item['context'],
            padding="max_length",
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'start_positions': torch.tensor(item['start_position']),
            'end_positions': torch.tensor(item['end_position']),
        }
