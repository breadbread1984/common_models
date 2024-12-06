#!/usr/bin/python3

from datasets import load_dataset

def format_example(example):
  messages = list()
  messages.append({'role': 'system', 'content': '请根据提供的文档和问题，生成简洁明了的答案。你需要从多个文档中提取关键信息，并整合这些信息来回答问题。同时，提供关键句子作为支持证据。'})
  # format user message
  user_message = f"问题:{example['question']}\n\n上下文：\n"
  for index, (title, sentences) in enumerate(zip(example['context']['title'], example['context']['sentences'])):
    user_message += f"文档{index}: {title}\n\n"
    for sent_id, sentence in enumerate(sentences):
      user_message += f"\t句子{sent_id}: {sentence}\n\n"
  messages.append({'role': 'user', 'content': user_message})
  # format assistance message
  assistant_message = f"答案:{example['answer']}\n\n关键句子:"
  assistant_message += ','.join([f"文档{example['context']['title'].index(title)}句子{sent_id}" for sent_id, title in zip(example['supporting_facts']['sent_id'], example['supporting_facts']['title'])])
  messages.append({'role': 'assistant', 'content': assistant_message})
  # NOTE: field name refs to https://huggingface.co/docs/trl/main/en/dataset_formats
  return {'messages': messages}

def load_hotpotqa():
  train = load_dataset('hotpotqa/hotpot_qa', 'distractor', split = 'train', trust_remote_code = True)
  valid = load_dataset('hotpotqa/hotpot_qa', 'distractor', split = 'validation', trust_remote_code = True)
  train = train.map(format_example)
  valid = valid.map(format_example)
  return train, valid

if __name__ == "__main__":
  train, valid = load_hotpotqa()
  print(valid[0]['messages'])
