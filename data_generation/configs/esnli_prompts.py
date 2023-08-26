prompt_with_examples = """\
Classify the relationship between two sentences: a premise and a hypothesis.

Assign one of three labels:
Entailment: The hypothesis is a logical inference that can be derived from the premise.
Contradiction: The hypothesis contradicts the information in the premise.
Neutral: The hypothesis neither logically follows from nor contradicts the premise.

Provide a brief explanation up to 30 words to justify your decision, then add a classification label.

Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```Two woman are holding packages.```
Response: ```Saying the two women are holding packages is a way to paraphrase that the packages they are holding are to go packages. #### Entailment```

Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```The sisters are hugging goodbye while holding to go packages after just eating lunch.```
Response: ```The to go packages may not be from lunch. #### Neutral```

Premise: ```Two women are embracing while holding to go packages.```
Hypothesis: ```The men are fighting outside a deli.```
Response: ```In the first sentence there is an action of affection between women while on the second sentence there is a fight between men. #### Contradiction```

Premise: ```{premise}```
Hypothesis: ```{hypothesis}```
Response: """

prompt_without_examples = """\
Classify the relationship between two sentences: a premise and a hypothesis.

Assign one of three labels:
Entailment: The hypothesis is a logical inference that can be derived from the premise.
Contradiction: The hypothesis contradicts the information in the premise.
Neutral: The hypothesis neither logically follows from nor contradicts the premise.

Provide a brief explanation up to 30 words to justify your decision, then add a classification label.

Premise: ```{premise}```
Hypothesis: ```{hypothesis}```
Response: """


if __name__ == "__main__":
    import json
    import os
    prompt_list = [prompt_with_examples] 
    with open(os.path.dirname(os.path.abspath(__file__)) + "/esnli_prompt_with_examples.json", "w") as f:
        json.dump(prompt_list, f)
    prompt_list.clear()
    prompt_list.append(prompt_without_examples)
    with open(os.path.dirname(os.path.abspath(__file__)) + "/esnli_prompt.json", "w") as f:
        json.dump(prompt_list, f)
