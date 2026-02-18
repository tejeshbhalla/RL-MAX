import torch
import deepspeed
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_PATH = "/workspace/rl-max/checkpoints/policy"
DS_CONFIG = "/workspace/rl-max/config/ds_config.json"



class GrpoTrainer:
    def __init__(self,model_path:str,ds_config:str,lr:float=1e-6,clip_eps=0.2,group_size:int=8):
        self.model_path = model_path
        self.ds_config = ds_config
        self.lr = lr
        self.clip_eps = clip_eps
        self.group_size = group_size
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model.gradient_checkpointing_enable()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)
        model_engine, optimizer, _, _ = deepspeed.initialize(
            model=self.model,
            optimizer=optimizer,
            config=ds_config,
        )
        print(f"DeepSpeed engine ready | device: {model_engine.device} | dtype: {model_engine.dtype}")
        print(f"ZeRO stage: {model_engine.zero_optimization_stage()}")
        self.model_engine = model_engine

    def prepare_inputs(self,prompts:list[str],completions:list[str]):

        device = self.model_engine.device
        batch_input_ids = []
        batch_completion_mask = []

        for prompt,completion in zip(prompts,completions):
            full_ids = self.tokenizer(prompt+completion,add_special_tokens=False).input_ids
            prompt_ids = self.tokenizer(prompt,add_special_tokens=False).input_ids     
            completion_mask = [0]* len(prompt_ids) + [1] * (len(full_ids)-len(prompt_ids))
            batch_input_ids.append(full_ids)
            batch_completion_mask.append(completion_mask)

        max_len_sequence  = max([len(ids) for ids in batch_input_ids])
        pad_token_id = self.tokenizer.pad_token_id

        padded_input_ids = []
        padded_completion_mask = []
        padded_attention_mask = []
        for ids,comp_mask in zip(batch_input_ids,batch_completion_mask):
            pad_len = max_len_sequence - len(ids)
            pad_ids = ids + [pad_token_id] * pad_len
            pad_comp_mask = comp_mask + [0] * pad_len 
            pad_attn_mask = [1] * len(ids) + [0] * pad_len 
            padded_input_ids.append(pad_ids)
            padded_completion_mask.append(pad_comp_mask)
            padded_attention_mask.append(pad_attn_mask)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long, device=device),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long, device=device),
            "completion_mask": torch.tensor(padded_completion_mask, dtype=torch.float, device=device)
        }
    def compute_token_log_probs(self,input_ids:torch.Tensor,attention_mask:torch.Tensor):
        output = self.model_engine(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            use_cache=False

        )
        logits  = output.logits  #bs,seq_len,vocab_size
        log_probs = torch.log_softmax(logits,dim=-1) #bs,seq_len,vocab_size
        
        shift_logits = log_probs[:,:-1,:]
        shift_labels = input_ids[:,1:] #bs,seq_len,1

        token_log_probs = torch.gather(shift_logits,dim=-1,index = shift_labels.unsqueeze(-1)).squeeze(-1)

        return token_log_probs

    def sequence_log_probs(self,log_probs: torch.Tensor, completion_mask:torch.Tensor)->torch.Tensor:
        shift_mask = completion_mask[:,1:]
        return (log_probs*shift_mask).sum(dim=-1)

    def compute_advantages(self,rewards:torch.Tensor)->torch.Tensor:
        grouped = rewards.view(-1,self.group_size)#bs,gs
        mean  = grouped.mean(dim=1,keepdims=True) #bs,1
        std = grouped.std(dim=1,keepdims=True)
        adv = (grouped-mean)/(std+1e-6) #bs,gs
        return adv.view(-1)

    def grpo_loss(self,new_seq_lp:torch.Tensor,old_seq_lp:torch.Tensor,advantages:torch.Tensor)->torch.Tensor:
        #shapes are  bs, old_seq_lp bs,  advantages bs
        ratio = torch.exp(new_seq_lp-old_seq_lp)
        clipped  = torch.clamp(ratio,1-self.clip_eps,1+self.clip_eps)
        return -torch.min(ratio * advantages,clipped*advantages).mean()

    def train_step(self,prompts:list[str],completions:list[str],old_log_probs:list[list[float]],rewards:list[float]):
        #everything here is of shape bs except log prob which is bs,seq_len 
        assert len(prompts) == len(completions) == len(old_log_probs) == len(rewards)
        assert len(prompts) % self.group_size == 0

        inputs = self.prepare_inputs(prompts,completions)
        input_ids = inputs['input_ids']
        attention_mask = inputs['attention_mask']
        completion_mask = inputs['completion_mask']

        old_log_prob_sum = torch.tensor([[sum(lp) for lp in old_log_probs]],dtype=torch.float,device=input_ids.device)

        rewards_t = torch.tensor(rewards,dtype=torch.float,device=input_ids.device)
        advantages = self.compute_advantages(rewards_t)

        new_token_lp = self.compute_token_log_probs(input_ids,attention_mask)
        new_seq_lp = self.sequence_log_probs(new_token_lp,completion_mask)

        loss = self.grpo_loss(new_seq_lp,old_log_prob_sum,advantages)
        self.model_engine.backward(loss)
        self.model_engine.step()


        return  {
        "loss": loss.detach().item(),
        "reward_mean": rewards_t.mean().item(),
        "reward_std": rewards_t.std(unbiased=False).item(),
        "adv_mean": advantages.mean().item(),
        "adv_std": advantages.std(unbiased=False).item(),
    }

    def save_checkpoint(self):
        if self.model_engine.local_rank == 0:
            self.model_engine.module.save_pretrained(self.model_path)
            self.tokenizer.save_pretrained(self.model_path)
            self.model_engine.module.gradient_checkpointing_enable()
            print(f"Checkpoint saved to {self.model_path}")


    
