import multiprocess 
import subprocess 
import argparse
import os

def run_scoring_responses(device_id, expansion, input_file, output_file):
    # Set cuda device id and innitialize new process and run scoring_responses.py
    process = subprocess.Popen(f'python scoring_responses.py ' \
                                    f'--device_id {device_id} ' \
                                    f'--input_file {input_file} ' \
                                    f'--expansion {expansion} ' \
                                    f'--output_file {output_file} ', 
                               shell=True)
    process.wait()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameters')
    parser.add_argument('--num_process', type=int, default=1)
    parser.add_argument('--expansion', type=int, required=True)
    args = parser.parse_args()
    
    for i in range(args.num_process):
        input_file = f'../generated_data/beam4_{i}.json'
        output_file = f'../generated_data/scored_beam4_{i}.json'
        process = multiprocess.Process(target=run_scoring_responses, args=(i, args.expansion, input_file, output_file))
        process.start()
    
    