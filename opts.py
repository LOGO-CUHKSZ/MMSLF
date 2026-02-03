import argparse

def parse_opts():
    parser = argparse.ArgumentParser()
    arguments = {
        'dataset': [
            dict(name='--datasetName',          # modify
                 type=str,
                 default="",
                 help=' '),
            dict(name='--dataPath',
                 default="",
                 type=str,
                 help=''),
            dict(name='--seq_lens',        # modify
                 default=[50, 50, 50],
                 type=list,
                 help=' '),
            dict(name='--feature_dims',         # modify
                 default=[768, 5, 20],
                 type=list,
                 help=' '),
            dict(name='--num_classes',
                 default=3,
                 type=int,
                 help=' '),
            dict(name='--KeyEval',
                 default="Loss",
                 type=str,
                 help=' '),
            dict(name='--use_only_one_bert',
                 action='store_true',
                 help=''),
            dict(name='--bert_finetune',
                 action='store_true',
                 help=''),
            dict(name='--prompt_bert_finetune',
                 action='store_true',
                 help=''),
            dict(name='--train_mode',
                 default="regression",
                 type=str,
                 help=' '),
            dict(name='--batch_size',
                 default=1,
                 type=int,
                 help=' '),
            dict(name='--min_sampling_rate',
                 default=0.5,
                 type=float,
                 help=' '),
            dict(name='--max_sampling_rate',
                 default=1.0,
                 type=float,
                 help=' '),
            dict(name='--num_workers',
                 default=8,
                 type=int,
                 help=' '),
        ],
        'network': [
            dict(name='--transformer_depth',
                 default=2,
                 type=int),
            dict(name='--crosstransformer_depth',
                 default=6,
                 type=int),
            dict(name='--CUDA_VISIBLE_DEVICES',          # modify
                 default='6',
                 type=str),
            dict(name='--fusion_layer_depth',
                 default=4,
                 type=int),
            dict(
                name='--encoder_depth',
                default=2,
                type=int,
                help='',
            ),
          dict(
               name='--fusion_depth',
               default=2,
               type=int,
               help='',
          )
        ],

        'common': [
            dict(name='--teacher_project_name',         # modify
                 default=None,
                 type=str
                 ),
            dict(name='--project_name',         # modify
                 default='RML_Set1',
                 type=str
                 ),
            dict(name='--seed',  # modify
                 default=1111,
                 type=int
                 ),
            dict(name='--use_cuda',
                 action='store_true',
                 default=False,
                 help='only cuda supported!'
                 ),
            dict(name='--ckpt_key',  # modify
                 default='mae',
                 type=str,
                 required=False
                 ),
            dict(name='--debug',
                 default=False,
                 action='store_true'),
            dict(
                name='--n_threads',
                default=3,
                type=int,
                help='Number of threads for multi-thread loading',
            ),
            dict(
                 name='--lr',
                 type=float,
                 default=5e-5), # modify
            dict(
                 name='--weight_decay',
                 type=float,
                 default=1e-4),
            dict(
                name='--n_epochs',
                default=100,
                type=int,
                help='Number of total epochs to run',
            ),
            dict(name='--alpha',  # modify
                 default=1.0,
                 type=float
                 ),
            dict(name='--beta',  # modify
                 default=1.0,
                 type=float
                 ),
            dict(name='--gamma',  # modify
                 default=0.2,
                 type=float
                 )
        ]
    }

    for group in arguments.values():
        for argument in group:
            name = argument['name']
            del argument['name']
            parser.add_argument(name, **argument)

    args = parser.parse_args()
    return args

