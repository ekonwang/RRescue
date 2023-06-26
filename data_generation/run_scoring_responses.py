import multiprocess 
import subprocess 
import argparse
import os

def run_scoring_responses(device_id, input_file, output_file, dataset):
    # Set cuda device id and innitialize new process and run scoring_responses.py
    os.environ['CUDA_VISIBLE_DEVICES'] = str(device_id)
    process = subprocess.Popen(['python', 'scoring_responses.py', 
                                    '--input_file', input_file, 
                                    '--output_file', output_file, 
                                    '--dataset', dataset], 
                               shell=True)
    process.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--dataset', type=str, default='Dahoas/rm-static')
    parser.add_argument('--devices', type=str, default='0')
    args = parser.parse_args()
    
    device_list = [int(i) for i in args.devices.split(',')]
    assert len(device_list) == args.num_process
    
    for i in range(args.num_process):
        input_file = f'../generated_data/beam4_{i}.txt'
        output_file = f'../generated_data/beam4_{i}_scored.txt'
        process = multiprocess.Process(target=run_scoring_responses, args=(device_list[i], input_file, output_file, args.dataset))
        process.start()
    
    