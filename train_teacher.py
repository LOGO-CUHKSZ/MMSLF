import os
import torch
from opts import *
from transformers import BertTokenizer

opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")
# print(device)

import numpy as np
from tqdm import tqdm
from core.dataset import MMDataLoader
from core.scheduler import get_scheduler
from core.utils import AverageMeter, setup_seed, tokenize_with_sampling, sequence_input_sampling
from tensorboardX import SummaryWriter
from models.MMSLF import build_teacher
from core.metric import MetricsTop


best_results = {}

train_mae, val_mae = [], []

if opt.datasetName == 'sims':
    Tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
else:
    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    opt = parse_opts()
    if opt.seed != -1:
        setup_seed(opt.seed)
        print("seed is fixed {}".format(opt.seed))

    log_path = os.path.join(".", "log", opt.project_name, str(opt.seed))
    print("log_path :", log_path)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    ckpt_path = f'./ckpt/{opt.project_name}/{opt.datasetName}/teacher'
    print("ckpt_path :", ckpt_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    if opt.datasetName == 'sims':
        Teacher = build_teacher(opt, encoder_depth=6, fusion_depth=6).to(device)
    else:
        Teacher = build_teacher(opt, encoder_depth=6, fusion_depth=6).to(device)
    models = {'Teacher': Teacher}

    # Compute the number of parameters
    num_para = count_parameters(models['Teacher'])
    print('parameters_count:', num_para)

    dataLoader = MMDataLoader(opt)

    optimizer = torch.optim.AdamW(Teacher.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)

    loss_fn = torch.nn.L1Loss()

    metrics = MetricsTop().getMetics(opt.datasetName)
    writer = SummaryWriter(logdir=log_path)

    for epoch in range(1, opt.n_epochs+1):
        train(models, dataLoader['train'], optimizer, loss_fn, epoch, writer, metrics)
        evaluate(models, dataLoader['valid'], optimizer, loss_fn, epoch, writer, ckpt_path, metrics, opt)
        scheduler_warmup.step()
    models['Teacher'].load_state_dict(torch.load(f'{ckpt_path}/best_mae_seed_{opt.seed}.pth')['state_dict'])
    print('Test')
    evaluate(models, dataLoader['test'], optimizer, loss_fn, 1, writer, ckpt_path, metrics, opt)
    writer.close()


def train(models, train_loader, optimizer, loss_fn, epoch, writer, metrics):
    global train_mae, best_results
    train_pbar = enumerate(train_loader)
    losses = AverageMeter()

    y_pred, y_true = [], []

    models['Teacher'].train()
    for cur_iter, data in train_pbar:
        img, audio, text, prompt = data['vision'].to(device), data['audio'].to(device), data['text'], data['prompt']
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]
        
        with torch.no_grad():
            if opt.datasetName == 'sims':
                img = sequence_input_sampling(img, opt.min_sampling_rate, 1.0)
                audio = sequence_input_sampling(audio, opt.min_sampling_rate, 1.0)
                text = tokenize_with_sampling(text, 'max_length', True, 50, opt.min_sampling_rate, 1.0, Tokenizer, device)
            else:
                img = sequence_input_sampling(img, opt.min_sampling_rate, 1.0)
                audio = sequence_input_sampling(audio, opt.min_sampling_rate, 1.0)
                text = tokenize_with_sampling(text, 'max_length', True, 50, opt.min_sampling_rate, 1.0, Tokenizer, device)

            prompt = Tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)

        out_teacher = models['Teacher'](img, audio, text, prompt)

        loss = loss_fn(out_teacher['preds'], label)

        losses.update(loss.item(), batchsize)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        y_pred.append(out_teacher['preds'].cpu())
        y_true.append(label.cpu())
        
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    train_results = metrics(pred, true)
    print('Train Epoch: {}: '.format(epoch), train_results)

    writer.add_scalar('train/loss', losses.value_avg, epoch)



def evaluate(models, eval_loader, optimizer, loss_fn, epoch, writer, ckpt_path, metrics, opt):
    global max_acc, best_acc2, best_acc3, best_f1, best_acc2_has_0, best_f1_has_0, best_acc5, best_acc7, best_mae, best_corr
    global val_mae, best_results
    test_pbar = enumerate(eval_loader)

    losses = AverageMeter()
    y_pred, y_true = [], []

    models['Teacher'].eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, prompt = data['vision'].to(device), data['audio'].to(device), data['text'], data['prompt']
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            text = Tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors='pt').to(device)
            prompt = Tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)

            out_teacher = models['Teacher'](img, audio, text, prompt)

            loss = loss_fn(out_teacher['preds'], label)

            losses.update(loss.item(), batchsize)

            y_pred.append(out_teacher['preds'].cpu())
            y_true.append(label.cpu())

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    test_results = metrics(pred, true)
    
    states = {
        'epoch': epoch + 1,
        'state_dict': models['Teacher'].state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    
    # keyMetric is MAE. It is also feasible to switch to other metrics.
    if opt.datasetName == 'sims':
        if epoch == 1:
            best_results['mae'] = test_results['MAE']
            best_results['corr'] = test_results['Corr']
            best_results['acc5'] = test_results['Mult_acc_5']
            best_results['acc3'] = test_results['Mult_acc_3']
            best_results['acc2_has0'] = test_results['Mult_acc_2']
            best_results['f1_has0'] = test_results['F1_score']
        else:
            if test_results['MAE'] < best_results['mae']:
                best_results['mae'] = test_results['MAE']
                best_results['corr'] = test_results['Corr']
                best_results['acc5'] = test_results['Mult_acc_5']
                best_results['acc3'] = test_results['Mult_acc_3']
                best_results['acc2_has0'] = test_results['Mult_acc_2']
                best_results['f1_has0'] = test_results['F1_score']
                torch.save(states, f'{ckpt_path}/best_mae_seed_{opt.seed}.pth')
    else:
        if epoch == 1:
            best_results['acc7'] = test_results['Mult_acc_7']
            best_results['acc5'] = test_results['Mult_acc_5']
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
                best_results['acc7'] = test_results['Mult_acc_7']
                best_results['acc5'] = test_results['Mult_acc_5']
                torch.save(states, f'{ckpt_path}/best_mae_seed_{opt.seed}.pth')

    print('Eval Epoch: {}: '.format(epoch), test_results)
    print('Best: {}: '.format(epoch), best_results)


if __name__ == '__main__':
    main()
