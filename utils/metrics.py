import torch
from torchmetrics import Metric
import jiwer
import whisper
import string

def remove_punctuation(text):
    text = text.replace('<|endoftext|>', '')
    return ''.join([char for char in text if char not in string.punctuation]).lower()



class WERCalculator(Metric):
    def __init__(self):
        super().__init__()
        self.add_state("total_insertions", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total_deletions", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total_substitutions", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        self.add_state("total_references", default=torch.tensor(0, dtype=torch.float32), dist_reduce_fx="sum")
        
        
    def update(self, references, predictions):
        for ref, pred in zip(references, predictions):
            # Apply custom preprocessing to both reference and prediction
            ref_processed = remove_punctuation(ref)
            pred_processed = remove_punctuation(pred)

            error = jiwer.compute_measures(ref_processed, pred_processed)
            self.total_insertions += error['insertions']
            self.total_deletions += error['deletions']
            self.total_substitutions += error['substitutions']
            self.total_references += error['hits'] + error['substitutions'] + error['deletions']
        
    def compute(self):
        results = {}
        if self.total_references > 0:
            results['WER'] = (self.total_insertions + self.total_deletions + self.total_substitutions) / self.total_references
        else:
            results['WER'] = torch.tensor(0.0)
        results['Insertions'] = self.total_insertions
        results['Deletions'] = self.total_deletions
        results['Substitutions'] = self.total_substitutions
        return results

    def reset(self):
        self.total_insertions.zero_()
        self.total_deletions.zero_()
        self.total_substitutions.zero_()
        self.total_references.zero_()
