import os
import torch
from opts import *
from transformers import BertTokenizer

opt = parse_opts()
os.environ["CUDA_VISIBLE_DEVICES"] = opt.CUDA_VISIBLE_DEVICES
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")

import numpy as np
from tqdm import tqdm
from core.dataset import MMDataLoader
from core.losses import MultimodalLoss
from core.optimizer import get_optimizer
from core.scheduler import get_scheduler
from core.utils import setup_seed, get_best_results
from tensorboardX import SummaryWriter
from models.MMSLF import build_teacher, build_student
from core.metric import MetricsTop

best_results = {}

train_mae, val_mae = [], []

if opt.datasetName == 'sims':
    Tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
else:
    Tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def main():
    global best_results

    opt = parse_opts()
    if opt.seed != -1:
        setup_seed(opt.seed)
        print("seed is fixed {}".format(opt.seed))

    log_path = os.path.join(".", "log", opt.teacher_project_name, opt.project_name, str(opt.seed))
    print("log_path :", log_path)

    ckpt_root = f'ckpt/{opt.project_name}/{opt.datasetName}/{opt.seed}'
    if opt.teacher_project_name is None:
        teacher_ckpt_root = ckpt_root
    else:
        teacher_ckpt_root = f'ckpt/{opt.teacher_project_name}/{opt.datasetName}'
    
    ckpt_path = f'{ckpt_root}/student/{opt.alpha}_{opt.beta}_{opt.gamma}'
    print("ckpt_path :", ckpt_path)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    Teacher = build_teacher(opt, encoder_depth=6, fusion_depth=6).to(device)
    Teacher.load_state_dict(torch.load(f'{teacher_ckpt_root}/teacher/best_{opt.ckpt_key}_seed_{opt.seed}.pth')['state_dict'], strict=False)

    Student = build_student(opt, encoder_depth=opt.encoder_depth, fusion_depth=opt.fusion_depth).to(device)
    models = {'Teacher': Teacher, 'Student': Student}

    dataLoader = MMDataLoader(opt)

    optimizer = torch.optim.AdamW(Student.parameters(),
                                 lr=opt.lr,
                                 weight_decay=opt.weight_decay)

    scheduler_warmup = get_scheduler(optimizer, opt)

    loss_fn = MultimodalLoss(alpha=opt.alpha, beta=opt.beta, gamma=opt.gamma)

    metrics = MetricsTop().getMetics(opt.datasetName)
    writer = SummaryWriter(logdir=log_path)

    for epoch in range(1, opt.n_epochs+1):
        train(models, dataLoader['train'], optimizer, loss_fn, epoch, writer, metrics)
        evaluate(models, dataLoader['valid'], optimizer, loss_fn, epoch, writer, ckpt_path, metrics, opt)
        scheduler_warmup.step()
    models['Student'].load_state_dict(torch.load(f'{ckpt_path}/best_mae_seed_{opt.seed}.pth')['state_dict'])
    print('Test')
    evaluate(models, dataLoader['test'], optimizer, loss_fn, 1, writer, ckpt_path, metrics, opt)
    writer.close()


def train(models, train_loader, optimizer, loss_fn, epoch, writer, metrics):
    global train_mae, best_results
    train_pbar = enumerate(train_loader)
    loss_list = {}

    y_pred, y_true = [], []
    y_pred_teacher= []

    models['Student'].train()
    for cur_iter, data in train_pbar:
        img, audio, text, prompt = data['vision'].to(device), data['audio'].to(device), data['text'], data['prompt']
        label = data['labels']['M'].to(device)
        label = label.view(-1, 1)
        batchsize = img.shape[0]

        with torch.no_grad():
            text = Tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors='pt').to(device)
            prompt = Tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)
            out_teacher = models['Teacher'](img, audio, text, prompt)

        out_student = models['Student'](img, audio, text)
        loss = loss_fn(out_student, out_teacher, label)

        loss['loss'].backward()
        optimizer.step()
        optimizer.zero_grad()

        if cur_iter == 0:
            loss_list = loss
        else:
            for key in loss_list.keys():
                loss_list[key] += loss[key]

        y_pred.append(out_student['preds'].cpu())
        y_true.append(label.cpu())
        y_pred_teacher.append(out_teacher['preds'].cpu())
        
    pred, true = torch.cat(y_pred), torch.cat(y_true)
    pred_teacher = torch.cat(y_pred_teacher)

    teacher_results = metrics(pred_teacher, true)
    print('Teacher Epoch: {}: '.format(epoch), teacher_results)
    train_results = metrics(pred, true)
    print('Train Epoch: {}: '.format(epoch), train_results)

    for key in loss_list.keys():
        writer.add_scalar(f'train/{key}', loss_list[key]/(cur_iter + 1), epoch)


def evaluate(models, eval_loader, optimizer, loss_fn, epoch, writer, ckpt_path, metrics, opt, mode=None):
    global max_acc, best_acc2, best_acc3, best_f1, best_acc2_has_0, best_f1_has_0, best_acc5, best_acc7, best_mae, best_corr
    global val_mae, best_results
    test_pbar = enumerate(eval_loader)

    y_pred, y_true = [], []
    y_pred_teacher= []
    loss_list = {}

    models['Student'].eval()
    with torch.no_grad():
        for cur_iter, data in test_pbar:
            img, audio, text, prompt = data['vision'].to(device), data['audio'].to(device), data['text'], data['prompt']
            label = data['labels']['M'].to(device)
            label = label.view(-1, 1)
            batchsize = img.shape[0]

            text = Tokenizer(text, padding='max_length', truncation=True, max_length=50, return_tensors='pt').to(device)
            prompt = Tokenizer(prompt, padding='max_length', truncation=True, max_length=512, return_tensors='pt').to(device)

            out_teacher = models['Teacher'](img, audio, text, prompt)
            out_student = models['Student'](img, audio, text)

            loss = loss_fn(out_student, out_teacher, label)

            if cur_iter == 0:
                loss_list = loss
            else:
                for key in loss_list.keys():
                    loss_list[key] += loss[key]

            y_pred.append(out_student['preds'].cpu())
            y_true.append(label.cpu())
            y_pred_teacher.append(out_teacher['preds'].cpu())
    
    for key in loss_list.keys():
        loss_list[key] /= (cur_iter + 1)

    pred, true = torch.cat(y_pred), torch.cat(y_true)
    pred_teacher = torch.cat(y_pred_teacher)
    
    test_results = metrics(pred, true)
    teacher_results = metrics(pred_teacher, true)
    print('Test Teacher Epoch: {}: '.format(epoch), teacher_results)
    print('Test Epoch: {}: '.format(epoch), test_results)
    
    for key in loss_list.keys():
        writer.add_scalar(f'test/{key}', loss_list[key]/(cur_iter + 1), epoch)
    
    # best_results, is_updated = get_best_results(opt.datasetName, best_results, test_results, epoch)
    key_metric = 'mae'
    best_results, is_updated = get_best_results(opt.datasetName, best_results, test_results, epoch, key_metric)
    if is_updated:
        states = {'epoch': epoch,'state_dict': models['Student'].state_dict(),'optimizer': optimizer.state_dict()}
        torch.save(states, f'{ckpt_path}/best_{key_metric}_seed_{opt.seed}.pth')
    print('Best: {}: '.format(epoch), best_results)

if __name__ == '__main__':
    main()
