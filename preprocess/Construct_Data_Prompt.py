import numpy as np
import pickle
import pandas as pd


if __name__ == '__main__':
    data_path = './SIMS/unaligned_39.pkl'
    prompt_path = './SIMS/prompt_sims.csv'

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
        prompts = pd.read_csv(prompt_path)
        
        data['train']['prompt'] = []
        data_len = len(data['train']['id'])
        for i in range(data_len):
            id = data['train']['id'][i]
            video_id = id.split('$_$')[0]
            clip_id = int(id.split('$_$')[1])
            
            filtered_prompt = prompts[(prompts.iloc[:, 0] == video_id) & (prompts.iloc[:, 1] == int(clip_id))]
            prompt = filtered_prompt['prompt']._values[0]

            data['train']['prompt'].append(prompt)

        data['valid']['prompt'] = []
        data_len = len(data['valid']['id'])
        for i in range(data_len):
            id = data['valid']['id'][i]
            video_id = id.split('$_$')[0]
            clip_id = id.split('$_$')[1]
            
            filtered_prompt = prompts[(prompts.iloc[:, 0] == video_id) & (prompts.iloc[:, 1] == int(clip_id))]
            prompt = filtered_prompt['prompt']._values[0]

            data['valid']['prompt'].append(prompt)


        data['test']['prompt'] = []
        data_len = len(data['test']['id'])
        for i in range(data_len):
            id = data['test']['id'][i]
            video_id = id.split('$_$')[0]
            clip_id = id.split('$_$')[1]
            
            filtered_prompt = prompts[(prompts.iloc[:, 0] == video_id) & (prompts.iloc[:, 1] == int(clip_id))]
            prompt = filtered_prompt['prompt']._values[0]

            data['test']['prompt'].append(prompt)


    # save data
    new_data_path = './SIMS/unaligned_39_prompt.pkl'
    with open(new_data_path, 'wb') as f:
        pickle.dump(data, f)

