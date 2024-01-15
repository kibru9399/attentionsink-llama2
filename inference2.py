from typing import Optional
import torch
import time
from pathlib import Path
import json
from sentencepiece import SentencePieceProcessor
from tqdm import tqdm

from model import Transformer, ModelArgs
from utils import load_prompts

class LLaMA:
    def __init__(self, model: Transformer, tokenizer: SentencePieceProcessor, model_args: ModelArgs):
        self.model = model
        self.tokenizer = tokenizer
        self.args = model_args
    @staticmethod
    def build(checkpoints_dir: str, tokenizer_path: str, load_model: bool, max_seq_len: int, max_batch_size: int, device: str):
        prev_time = time.time()
        if load_model:
            checkpoints = sorted(Path(checkpoints_dir).glob('*.pth'))
            assert len(checkpoints) > 0, 'No checkpoints found in the folder'
            chk_path = checkpoints[0]
            print(f'Loading checkpoints{chk_path}')
            checkpoint = torch.load(chk_path, map_location='cpu')
            print(f'Loaded checkoint in{(time.time() - prev_time):.2f}s')
            prev_time = time.time()

        with open(Path(checkpoints_dir) / 'params.json', 'r') as f:
            params = json.loads(f.read())
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            device=device,
            **params
        )

        tokenizer = SentencePieceProcessor()
        tokenizer.load(tokenizer_path)
        model_args.vocab_size = tokenizer.vocab_size()

        if device == 'cuda':
            torch.set_default_tensor_type(torch.cuda.HalfTensor)
        else:
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        model = Transformer(model_args).to(device)
        if load_model:
            del checkpoint['rope.freqs']
            model.load_state_dict(checkpoint, strict=True)
            print(f'loaded state dict in {(time.time() - prev_time):.2f}s')

        return LLaMA(model, tokenizer, model_args)
    def text_completion(self, prompts: list[str], temperature: float = 0.6, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
            max_gen_len = self.args.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(prompt, out_type=int, add_bos=True,add_eos=False) for prompt in prompts]
        batch_size = len(prompt_tokens)
        assert batch_size <= self.args.max_batch_size, 'batch size too large'
        max_prompt_len = max(len(prompt) for prompt in prompt_tokens)
        assert max_prompt_len <= self.args.max_seq_len, 'check the seq len'
        total_len = min(self.args.max_seq_len, max_gen_len + max_prompt_len)
        pad_id = self.tokenizer.pad_id()
        # create a list that will contain the generated tokens
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)

        for k, t in enumerate(prompt_tokens):
            tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
        eos_reached = torch.tensor([False]*batch_size, device=device)
        prompt_tokens_mask = tokens != pad_id #true if the token is prompt token, false otherwise

        for cur_pos in tqdm(range(1, total_len), desc='Generating tokens'):
            with torch.no_grad():
                logits = self.model.forward(tokens[:, cur_pos-1:cur_pos],  cur_pos)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                next_token = self._sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            next_token = next_token.reshape(-1)
            next_token = torch.where(prompt_tokens_mask[:, cur_pos], tokens[:, cur_pos], next_token)
            tokens[:, cur_pos] = next_token
            # if it's not a prompt token and the predicted prompt is EOS token
            eos_reached |= (~prompt_tokens_mask[:, cur_pos]) & (next_token == self.tokenizer.eos_id)
            if all(eos_reached):
                break
        out_tokens = []
        out_text = []
        for prompt_index, current_prompt_tokens in enumerate(tokens.tolist()):
            if self.tokenizer.eos_id in current_prompt_tokens:
                eos_idx = current_prompt_tokens.index(self.tokenizer.eos_id)
                current_prompt_tokens = current_prompt_tokens[:eos_idx]
            out_tokens.append(current_prompt_tokens)
            out_text.append(self.tokenizer.decode(current_prompt_tokens))
        # clean the kv cache for the next round of prompting
        return (out_tokens, out_text)
    def _sample_top_p(self, probs, p):
        #probs = probs.squeeze(1)
        probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
        probs_sum = torch.cumsum(probs_sort,dim=-1)
        mask = probs_sum - probs_sort > p
        probs_sort[mask] = 0.0
        probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
        next_token = torch.multinomial(probs_sort, num_samples=1)
        next_token = torch.gather(probs_idx, -1, next_token)
        return next_token

    def text_completion2(self, prompt: list[str], temperature: float = 0.0, top_p: float = 0.9, max_gen_len: Optional[int] = None):
        if max_gen_len is None:
          max_gen_len = self.args.max_seq_len - 1
        prompt_tokens = self.tokenizer.encode(prompt, out_type=int, add_bos=True,add_eos=False)
        prompt_len = len(prompt_tokens)
        batch_size = 1
        assert prompt_len < self.args.max_seq_len,'too large of a prompt'
        total_len = min(self.args.max_seq_len, prompt_len + max_gen_len)
        pad_id = self.tokenizer.pad_id()
        tokens = torch.full((batch_size, total_len), pad_id, dtype=torch.long, device=device)
        tokens[:, :prompt_len] = torch.tensor(prompt_tokens, dtype=torch.long, device=device)
        eos_reached = torch.tensor([False]*batch_size, device=device)
        prompt_token_mask = tokens != pad_id
        generated_ids = []
        input_token = tokens[:, :1]
        pos = 0

        for cur_pos in range(1, total_len):

          with torch.no_grad():
            logits = self.model.forward(input_token, cur_pos)
          if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = self._sample_top_p(probs, top_p)
          else:
            next_token = torch.argmax(logits[:, -1], dim=-1).view(batch_size, -1)
          #nexxt token to be used
          if cur_pos <= prompt_len:
            input_token = tokens[:, cur_pos-1:cur_pos]
          else:
            input_token = next_token

          next_token = next_token.reshape(-1)

          if cur_pos > prompt_len:
            generated_ids.append(next_token.item())

          generated_text = (
                  self.tokenizer.decode(
                      generated_ids,
                  )
                  .strip()
                  .split(" ")
              )

          now = len(generated_text) - 1
          if now > pos:
              print(" ".join(generated_text[pos:now]), end=" ", flush=True)
              pos = now

          if next_token.item() == self.tokenizer.eos_id():
              break








if __name__ == '__main__':
    torch.manual_seed(0)
    allow_cuda = True
    device = 'cuda' if torch.cuda.is_available() and allow_cuda else 'cpu'

     

    model = LLaMA.build(
        checkpoints_dir ='llama-2-7b-chat/', 
        tokenizer_path = 'tokenizer.model', 
        load_model=True, 
        max_seq_len=1024, 
        max_batch_size=1, 
        device=device
    )
    print('ALL OK!')

    prompts = load_prompts()

    for i in range(len(prompts)):
       for i in range(len(prompts)):
        print(prompts[i]) 
        model.text_completion2(prompts[i], max_gen_len=500, temperature=0) 
        print('-'*50)
        #clean the kv cache for the next prompt
        for j in range(32):
            model.model.layers[j].attention.clean_kv_cache(500+len(prompts[i+1]))