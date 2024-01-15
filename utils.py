import json
def load_jsonl(
    file_path,
):
    list_data_dict = []
    with open(file_path, "r") as f:
        for line in f:
            list_data_dict.append(json.loads(line))
    return list_data_dict
def load_prompts():
    list_data = load_jsonl("/content/drive/MyDrive/Attention Sink/question.jsonl")
    prompts = []
    for sample in list_data:
        prompts += sample["turns"]
    return prompts