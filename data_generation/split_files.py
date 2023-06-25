from datasets import load_dataset
import argparse
import json
import sys
import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='Dahoas/rm-static')
    args = parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    dataset = load_dataset(args.dataset)['train']
    with open(sys.argv[1] + '/raw_generation.json', 'r') as f:
        samples = [json.loads(item) for item in f.readlines()]
    samples = samples[:len(dataset)]

    print('=='*10)
    print(samples[-1])
    print(dataset[-1])

    buffer = []
    count = 0
    split_size = (len(samples) + args.num_process - 1) // args.num_process
    for idx in tqdm.tqdm(range(len(samples))):
        temp = [samples[idx][0][0], [item[1] for item in samples[idx]]]
        if args.dataset == 'Dahoas/rm-static':
            temp.append(dataset[idx]['chosen'])
            temp.append(dataset[idx]['rejected'])
        
        temp[1] = [i.replace(temp[0], "") for i in temp[1]]
        buffer.append(temp)
        if len(buffer) == split_size:
            with open(sys.argv[2] + f'/beam4_{count}.txt', 'w') as f:
                for item in buffer:
                    f.write(json.dumps(item) + '\n')
            count += 1
            buffer = []

    if len(buffer):
        with open(sys.argv[2] + f'/beam4_{count}.txt', 'w') as f:
            for item in buffer:
                f.write(json.dumps(item) + '\n')
        


        