"""Dataloader for text files."""
import json

import torch
import torch.utils.data as data
import transformers as tf

class TextFileDataset(data.IterableDataset):
    """Text file dataset."""
    def __init__(self, file_path, model_card, max_len, eval_mode=False, dummy_mode=False):
        super().__init__()
        self.file_path = file_path
        self.model_card = model_card
        self.max_len = max_len
        self.eval_mode = eval_mode
        self.dummy_mode = dummy_mode

        self.tokenizer = tf.AutoTokenizer.from_pretrained(self.model_card)

    def __iter__(self):
        counter = 0
        with open(self.file_path, 'r') as f: # pylint: disable=invalid-name, unspecified-encoding
            for jsonl_line in f:
                text = json.loads(jsonl_line)['text']
                # Iterate over segments of length self.max_len * 2
                for i in range(0, len(text), self.max_len * 2):
                    line_segment = text[i:i + self.max_len * 2]
                    yield self.transform(line_segment)
                    counter += 1
                    if (counter >= 1000 and self.dummy_mode) or (
                            counter >= 10000 and self.eval_mode):
                        break
                    if counter % 1000 == 0:
                        print(f'Processed {counter} lines')
                if (counter >= 1000 and self.dummy_mode) or (counter >= 10000 and self.eval_mode):
                    break

    def transform(self, line):
        """Transforms the line into a tensor."""
        if self.eval_mode:
            return self.tokenizer.encode_plus(
                line, max_length=self.max_len, pad_to_max_length=True,
                return_tensors='pt', truncation=True)
        else:
            return self.tokenizer.encode_plus(
                line, max_length=self.max_len, pad_to_max_length=True,
                return_tensors='pt', truncation=True)

def collate(batch):
    """Collate the batch."""
    if all([item is None for item in batch]):
        return None
    geti = lambda i: [x[i] for x in batch if x is not None] # pylint: disable=unnecessary-lambda-assignment
    input_ids = torch.cat(geti('input_ids'), 0) # pylint: disable=no-member
    attention_mask = torch.cat(geti('attention_mask'), 0) # pylint: disable=no-member

    return {"input_ids": input_ids, "attention_mask": attention_mask}
