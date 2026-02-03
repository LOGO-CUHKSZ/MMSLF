import torch
from torch import nn
from .vit import Transformer, ModulateCrossTransformer
from .bert import BertTextEncoder


class Teacher(nn.Module):
    def __init__(self, dataset, encoder_depth, fusion_depth, bert_pretrained, use_only_one_bert, bert_finetune, prompt_bert_finetune):
        super(Teacher, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=bert_finetune, transformers='bert', pretrained=bert_pretrained)
        if use_only_one_bert:
            self.bertmodel_prompt = self.bertmodel
        else:
            self.bertmodel_prompt = BertTextEncoder(use_finetune=prompt_bert_finetune, transformers='bert', pretrained=bert_pretrained)

        self.dataset = dataset  

        if dataset == 'mosi':
            self.proj_l0 = nn.Linear(768, 64)
            self.proj_a0 = nn.Linear(5, 64)
            self.proj_v0 = nn.Linear(20, 64)
            self.proj_p0 = nn.Linear(768, 64)
            self.proj_p = Transformer(num_frames=512, save_hidden=False, token_len=50, dim=64, depth=2, heads=6, mlp_dim=128, dim_head = 32, dropout = 0.)
            self.l_a_encoder = ModulateCrossTransformer(source_num_frames=375, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=128, dim_head = 32, dropout = 0.5)
            self.l_v_encoder = ModulateCrossTransformer(source_num_frames=500, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=128, dim_head = 32, dropout = 0.5)
            self.fusion_encoder = Transformer(num_frames=100, save_hidden=False, token_len=4, dim=64, depth=fusion_depth, heads=6, mlp_dim=64, dim_head = 32, dropout = 0.)
        elif dataset == 'mosei':
            self.proj_l0 = nn.Linear(768, 64)
            self.proj_a0 = nn.Linear(74, 64)
            self.proj_v0 = nn.Linear(35, 64)
            self.proj_p0 = nn.Linear(768, 64)
            self.proj_p = Transformer(num_frames=512, save_hidden=False, token_len=50, dim=64, depth=2, heads=6, mlp_dim=128, dim_head = 32, dropout = 0.)
            self.l_a_encoder = ModulateCrossTransformer(source_num_frames=500, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=128, dim_head = 32, dropout = 0.5)
            self.l_v_encoder = ModulateCrossTransformer(source_num_frames=500, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=128, dim_head = 32, dropout = 0.5)
            self.fusion_encoder = Transformer(num_frames=100, save_hidden=False, token_len=4, dim=64, depth=fusion_depth, heads=6, mlp_dim=64, dim_head = 32, dropout = 0.)
        elif dataset == 'sims':
            self.proj_l0 = nn.Linear(768, 64)
            self.proj_a0 = nn.Linear(33, 64)
            self.proj_v0 = nn.Linear(709, 64)
            self.proj_p0 = nn.Linear(768, 64)
            self.proj_p = Transformer(num_frames=512, save_hidden=False, token_len=50, dim=64, depth=2, heads=8, mlp_dim=64, dim_head = 32, dropout = 0.)
            self.l_a_encoder = ModulateCrossTransformer(source_num_frames=400, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=768, dim_head = 32, dropout = 0.5)
            self.l_v_encoder = ModulateCrossTransformer(source_num_frames=55, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=768, dim_head = 32, dropout = 0.5)
            self.fusion_encoder = Transformer(num_frames=100, save_hidden=False, token_len=4, dim=64, depth=fusion_depth, heads=8, mlp_dim=64, dim_head = 32, dropout = 0.)
        else:
            assert False

        self.reg = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, video, audio, text, prompt):
        b = video.size(0)

        x_visual = self.proj_v0(video)
        x_audio = self.proj_a0(audio)
        x_text = self.bertmodel(text)
        x_text = self.proj_l0(x_text)

        x_prompt = self.bertmodel_prompt(prompt)
        x_prompt = self.proj_p0(x_prompt)
        x_prompt = self.proj_p(x_prompt)[:, :50]

        out_la = self.l_a_encoder(x_audio, x_text, x_prompt)
        h_la, h_la_attn, h_la_modulate_attn = out_la['out'], out_la['attn'], out_la['modulate_attn']

        out_lv = self.l_v_encoder(x_visual, x_text, x_prompt)
        h_lv, h_lv_attn, h_lv_modulate_attn = out_lv['out'], out_lv['attn'], out_lv['modulate_attn']

        h_m = torch.cat([h_la, h_lv], dim=1)
        h_reg = self.fusion_encoder(h_m)[:, :4]

        out_reg = self.reg(h_reg.reshape(b, -1))

        return {'preds': out_reg, 'h_la_attn': h_la_attn, 'h_lv_attn': h_lv_attn, 'h_la_modulate_attn': h_la_modulate_attn, 'h_lv_modulate_attn': h_lv_modulate_attn, 'h_la': h_la, 'h_lv': h_lv, 'h_reg': h_reg}


class Student(nn.Module):
    def __init__(self, dataset, encoder_depth, fusion_depth, bert_pretrained, bert_finetune):
        super(Student, self).__init__()

        self.bertmodel = BertTextEncoder(use_finetune=bert_finetune, transformers='bert', pretrained=bert_pretrained)
        self.dataset = dataset  

        if dataset == 'mosi':
            self.proj_l0 = nn.Linear(768, 64)
            self.proj_a0 = nn.Linear(5, 64)
            self.proj_v0 = nn.Linear(20, 64)
            self.l_a_encoder = ModulateCrossTransformer(source_num_frames=375, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=128, dim_head = 32, dropout = 0.5)
            self.l_v_encoder = ModulateCrossTransformer(source_num_frames=500, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=128, dim_head = 32, dropout = 0.5)
            self.fusion_encoder = Transformer(num_frames=100, save_hidden=False, token_len=4, dim=64, depth=fusion_depth, heads=6, mlp_dim=64, dim_head = 32, dropout = 0.)
        elif dataset == 'mosei':
            self.proj_l0 = nn.Linear(768, 64)
            self.proj_a0 = nn.Linear(74, 64)
            self.proj_v0 = nn.Linear(35, 64)
            self.proj_p0 = nn.Linear(768, 64)
            self.l_a_encoder = ModulateCrossTransformer(source_num_frames=500, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=128, dim_head = 32, dropout = 0.5)
            self.l_v_encoder = ModulateCrossTransformer(source_num_frames=500, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=128, dim_head = 32, dropout = 0.5)
            self.fusion_encoder = Transformer(num_frames=100, save_hidden=False, token_len=4, dim=64, depth=fusion_depth, heads=6, mlp_dim=64, dim_head = 32, dropout = 0.)
        elif dataset == 'sims':
            self.proj_l0 = nn.Linear(768, 64)
            self.proj_a0 = nn.Linear(33, 64)
            self.proj_v0 = nn.Linear(709, 64)
            self.l_a_encoder = ModulateCrossTransformer(source_num_frames=400, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=768, dim_head = 32, dropout = 0.5)
            self.l_v_encoder = ModulateCrossTransformer(source_num_frames=55, tgt_num_frames=50, dim=64, depth=encoder_depth, heads=4, mlp_dim=768, dim_head = 32, dropout = 0.5)
            self.fusion_encoder = Transformer(num_frames=100, save_hidden=False, token_len=4, dim=64, depth=fusion_depth, heads=4, mlp_dim=64, dim_head = 32, dropout = 0.)
        else:
            exit(0)

        self.reg = nn.Sequential(
            nn.Linear(256, 1)
        )

    def forward(self, video, audio, text):        
        b = video.size(0)

        x_visual = self.proj_v0(video)
        x_audio = self.proj_a0(audio)
        x_text = self.bertmodel(text)
        x_text = self.proj_l0(x_text)

        out_la = self.l_a_encoder(x_audio, x_text, None)
        h_la, h_la_attn, h_la_modulate_attn = out_la['out'], out_la['attn'], out_la['modulate_attn']

        out_lv = self.l_v_encoder(x_visual, x_text, None)
        h_lv, h_lv_attn, h_lv_modulate_attn = out_lv['out'], out_lv['attn'], out_lv['modulate_attn']

        h_m = torch.cat([h_la, h_lv], dim=1)
        h_reg = self.fusion_encoder(h_m)[:, :4]

        out_reg = self.reg(h_reg.reshape(b, -1))

        return {'preds': out_reg, 'h_la_attn': h_la_attn, 'h_lv_attn': h_lv_attn, 'h_la_modulate_attn': h_la_modulate_attn, 'h_lv_modulate_attn': h_lv_modulate_attn, 'h_la': h_la, 'h_lv': h_lv, 'h_reg': h_reg}



def build_teacher(opt, encoder_depth=6, fusion_depth=6):
    if opt.datasetName == 'sims':
        model = Teacher(dataset = opt.datasetName, \
                        encoder_depth=encoder_depth, \
                        fusion_depth=fusion_depth, \
                        bert_pretrained = 'bert-base-chinese', \
                        use_only_one_bert=opt.use_only_one_bert, \
                        bert_finetune = opt.bert_finetune, \
                        prompt_bert_finetune = opt.prompt_bert_finetune)
    else:
        model = Teacher(dataset = opt.datasetName, \
                        encoder_depth=encoder_depth, \
                        fusion_depth=fusion_depth, \
                        bert_pretrained = 'bert-base-uncased', \
                        use_only_one_bert=opt.use_only_one_bert, \
                        bert_finetune = opt.bert_finetune, \
                        prompt_bert_finetune = opt.prompt_bert_finetune)
    return model




def build_student(opt, encoder_depth=2, fusion_depth=2):
    if opt.datasetName == 'sims':
        model = Student(dataset = opt.datasetName, encoder_depth=encoder_depth, fusion_depth=fusion_depth, bert_pretrained = 'bert-base-chinese', bert_finetune = opt.bert_finetune)
    else:
        model = Student(dataset = opt.datasetName, encoder_depth=encoder_depth, fusion_depth=fusion_depth, bert_pretrained = 'bert-base-uncased', bert_finetune = opt.bert_finetune)
    return model