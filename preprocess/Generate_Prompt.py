import os
import numpy as np
import argparse
import csv
from tqdm import tqdm
import time
from openai import OpenAI
import base64

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def load_csv(csv_file):
    lines = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            lines.append(row)
    return lines


def check_if_exists(output_csv_file, video_id, clip_id):
    if not os.path.exists(output_csv_file):
        return False
    
    with open(output_csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['video_id'] == video_id and row['clip_id'] == clip_id:
                return True
    return False


def request_openai(client, question, base64_images, model='gpt-4o-mini'):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": question}
            ]
        }
    ]
    
    for img in base64_images:
        messages[0]["content"].append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{img}"
            }
        })
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=1000
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        time.sleep(5)  
        return None


def main():
    args = parse_args()

    if args.dataset == 'mosi':
        data_dir = './MOSI'
        csv_file = './MOSI/label.csv'
    elif args.dataset == 'sims':
        data_dir = './SIMS'
        csv_file = './SIMS/label.csv'
    elif args.dataset == 'mosei':
        data_dir = './MOSEI'
        csv_file = './MOSEI/label.csv'
    else:
        assert False, "Dataset not found!"

    output_csv_file = f'./prompt_{args.dataset}.csv'

    client = OpenAI(
        base_url='placeholder',
        api_key='placeholder'
    )
    
    lines = load_csv(csv_file)

    count = 0
    for iter, line in tqdm(enumerate(lines), total=len(lines)):
        base64_image = []

        video_id = line['video_id']
        clip_id = line['clip_id']
        text = line['text']
        label = line['label']
        mode = line['mode']
        
        if check_if_exists(output_csv_file, video_id, clip_id):
            continue

        video_path = os.path.join(data_dir, video_id, clip_id, 'frames')
        if os.path.exists(video_path):
            # Uniform Sampling
            frames = os.listdir(video_path)
            frames.sort()
            n_frames = len(frames)
            if n_frames < 3:
                assert False, "Number of frames is less than 3!"
            idxs = np.linspace(0, n_frames-1, 3, dtype=int)
            frames = [os.path.join(video_path, frames[idx]) for idx in idxs]

            for frame in frames:
                base64_image.append(encode_image(frame))

        if mode != 'train':
            label = None

        question = f"""### Background ###
You are a multimodal sentiment analysis expert. We provide you with a video-text pair and corresponding sentiment label. Please provide detailed hints to help task-specific small models identify sentiment cues.

### Video-text Pair and Corresponding Label ###
Video: Please refer to the video input.
Text: {text}
Label: {label}
Note: Only the labels of the training set data are not None.

### Response Requirements ###
1. Focus on facial expressions in the video, especially smiles, frowns, and eye movements.
2. Focus on the language in the video, especially emotionally charged words and phrases.

### Output Format ###
Your output should consists of the following parts:
Your output consists of the following parts:
1. Visual cues: Observations related to facial expressions in the video.
2. Language cues: Important sentiment cues found in the provided text.
"""

        response = request_openai(client, question, base64_image, model='gpt-4o-mini')
        
        if response is None:
            continue
            
        print(f"Processed {count + 1} samples")

        if not os.path.exists(output_csv_file):
            flag = 1
        else:
            flag = 0

        with open(output_csv_file, 'a+', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if flag:
                writer.writerow(["video_id", "clip_id", "label", "mode", "prompt"])
            writer.writerow([video_id, clip_id, label, mode, response])
        count += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--dataset", type=str, default="sims")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main()