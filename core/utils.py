import torch
import numpy as np
import random


class AverageMeter(object):
    def __init__(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def reset(self):
        self.value = 0
        self.value_avg = 0
        self.value_sum = 0
        self.count = 0

    def update(self, value, count):
        self.value = value
        self.value_sum += value * count
        self.count += count
        self.value_avg = self.value_sum / self.count


def save_model(save_path, epoch, model, optimizer):
    states = {
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    torch.save(states, save_path)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def tokenize_with_sampling(prompt, padding, truncation, max_length, min_sample_rate, max_sample_rate, tokenizer, device):
    ret = tokenizer(
        prompt,
        padding=padding,
        truncation=truncation,
        max_length=max_length,
        return_tensors='pt'
    )

    input_ids, attention_mask, token_type_ids = ret['input_ids'], ret['attention_mask'], ret['token_type_ids']
    unk_token_id = tokenizer.unk_token_id

    sample_rates = np.random.uniform(min_sample_rate, max_sample_rate, size=input_ids.shape[0])
    masks = ~(torch.rand(input_ids.shape) < torch.tensor(sample_rates).unsqueeze(1))
    input_ids = torch.where(masks, unk_token_id, input_ids)

    tokenized_input = {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'token_type_ids': token_type_ids.to(device)
    }

    return tokenized_input



def sequence_input_sampling(input_sequence, min_sample_rate, max_sample_rate):
    sample_rates = np.random.uniform(min_sample_rate, max_sample_rate, size=input_sequence.shape[0])
    masks = ~(torch.rand(input_sequence.shape, device=input_sequence.device) < torch.tensor(sample_rates, device=input_sequence.device).unsqueeze(1).unsqueeze(1))
    input_sequence = torch.where(masks, torch.tensor(0.), input_sequence)

    return input_sequence



def get_best_results(datasetName, best_results, test_results, epoch, key_metric):
    is_updated = False
    if key_metric == 'mae':
        if datasetName == 'sims':
            if epoch == 1:
                best_results['mae'] = test_results['MAE']
                best_results['corr'] = test_results['Corr']
                best_results['acc2_has0'] = test_results['Mult_acc_2']
                best_results['f1_has0'] = test_results['F1_score']
            else:
                if test_results['MAE'] < best_results['mae']:
                    best_results['mae'] = test_results['MAE']
                    best_results['corr'] = test_results['Corr']
                    best_results['acc2_has0'] = test_results['Mult_acc_2']
                    best_results['f1_has0'] = test_results['F1_score']
                    is_updated = True
        else:
            if epoch == 1:
                best_results['acc2_has0'] = test_results['Has0_acc_2']
                best_results['f1_has0'] = test_results['Has0_F1_score']
                best_results['acc2_non0'] = test_results['Non0_acc_2']
                best_results['f1_non0'] = test_results['Non0_F1_score']
                best_results['mae'] = test_results['MAE']
                best_results['corr'] = test_results['Corr']
            else:
                if test_results['MAE'] < best_results['mae']:
                    best_results['mae'] = test_results['MAE']
                    best_results['corr'] = test_results['Corr']
                    best_results['acc2_has0'] = test_results['Has0_acc_2']
                    best_results['f1_has0'] = test_results['Has0_F1_score']
                    best_results['acc2_non0'] = test_results['Non0_acc_2']
                    best_results['f1_non0'] = test_results['Non0_F1_score']
                    is_updated = True
    
    elif key_metric == 'acc':
        if datasetName == 'sims':
            if epoch == 1:
                best_results['mae'] = test_results['MAE']
                best_results['corr'] = test_results['Corr']
                best_results['acc2_has0'] = test_results['Mult_acc_2']
                best_results['f1_has0'] = test_results['F1_score']
            else:
                if test_results['Mult_acc_2'] > best_results['acc2_has0']:
                    best_results['mae'] = test_results['MAE']
                    best_results['corr'] = test_results['Corr']
                    best_results['acc2_has0'] = test_results['Mult_acc_2']
                    best_results['f1_has0'] = test_results['F1_score']
                    is_updated = True
        else:
            if epoch == 1:
                best_results['acc2_has0'] = test_results['Has0_acc_2']
                best_results['f1_has0'] = test_results['Has0_F1_score']
                best_results['acc2_non0'] = test_results['Non0_acc_2']
                best_results['f1_non0'] = test_results['Non0_F1_score']
                best_results['mae'] = test_results['MAE']
                best_results['corr'] = test_results['Corr']
            else:
                if test_results['Mult_acc_2'] > best_results['acc2_has0']:
                    best_results['mae'] = test_results['MAE']
                    best_results['corr'] = test_results['Corr']
                    best_results['acc2_has0'] = test_results['Has0_acc_2']
                    best_results['f1_has0'] = test_results['Has0_F1_score']
                    best_results['acc2_non0'] = test_results['Non0_acc_2']
                    best_results['f1_non0'] = test_results['Non0_F1_score']
                    is_updated = True
    else:
        assert False

    return best_results, is_updated