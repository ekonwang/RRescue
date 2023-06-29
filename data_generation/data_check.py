import json

if __name__ == '__main__':
    file = '/workspace/RL/generated_data/scored_beam4_0.json'
    with open(file, 'r') as f:
        samples = json.load(f)
    hit = 0
    tot = len(samples)
    for idx, sample in enumerate(samples):
        labels = ['entailment', 'neutral', 'contradiction']
        score_dict = {
            'entailment': 0,
            'neutral': 0,
            'contradiction': 0
        }
        responses = sample['response']
        label = sample['label']
        scores = sample['scores']
        predicted_label_formax = None
        # 每组两个句子 beam_size = 2
        for i in range(len(scores)):
            candidate = labels[i//2]
            score_dict[candidate] += scores[i]
        
        max_score = 0
        for candidate in ['entailment', 'neutral', 'contradiction']:
            if score_dict[candidate] > max_score:
                max_score = score_dict[candidate]
                predicted_label_formax = candidate
        if predicted_label_formax != label:
            print(idx, predicted_label_formax, label)
        else:
            hit += 1
    print(score_dict)
    print('*'*10, f'{hit/tot*100:.2f}%({hit}/{tot})')