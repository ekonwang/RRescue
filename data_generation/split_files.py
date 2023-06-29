from datasets import load_dataset
import argparse
import json
import sys
import tqdm

map_dict = {
    0: 'entailment',
    1: 'neutral',
    2: 'contradiction'
}

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--out_path', type=str, default='../generated_data')
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='Dahoas/rm-static')
    parser.add_argument('--expansion', type=int, default=1, required=True)
    args = parser.parse_args()
    
    if args.expansion > 1:
        args.expansion = 3
    else:
        args.expansion = 1
    return args


def post_process(responses):
    for i, response in enumerate(responses):
        response = response.strip().lstrip('\n')
        response = response.split('\n')[0]
        response = response.split('.', 1)[1]
        responses[i] = response
    return responses


if __name__ == '__main__':
    args = parse_args()
    dataset = load_dataset(args.dataset)['train']
    dataset_name = args.dataset.split('/')[-1]
    with open(args.out_path + f'/raw_generation_{dataset_name}.json', 'r') as f:
        samples = json.load(f)
    
    print('=='*10)
    print(samples[-1])
    print(dataset[-1])

    buffer = []
    count = 0
    split_size = (len(samples) + args.num_process - 1) // args.num_process
    for idx in tqdm.tqdm(range(len(samples) // args.expansion)):
        for inner in range(args.expansion):
            temp = [samples[idx * args.expansion + inner][0][0], [item[1] for item in samples[idx * args.expansion + inner]]]
            if args.dataset == 'Dahoas/rm-static':
                temp.append(dataset[idx]['chosen'])
                temp.append(dataset[idx]['rejected'])
            if args.dataset == 'esnli':
                temp[1] = post_process(temp[1])
                temp.append(dataset[idx]['explanation_1'])
                label = dataset[idx]['label']
                temp.append(map_dict[label])
            
            temp[1] = [i.replace(temp[0], "") for i in temp[1]]
            buffer.append(temp)
            if len(buffer) == split_size:
                with open(args.out_path + f'/beam4_{count}.json', 'w') as f:
                    json.dump(buffer, f, indent=4)
                count += 1
                buffer = []

    if len(buffer):
        with open(args.out_path + f'/beam4_{count}.json', 'w') as f:
            json.dump(buffer, f, indent=4)
