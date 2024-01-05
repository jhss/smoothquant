import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
from smoothquant.observer import OutlierObserver
from datasets import load_dataset
from torch.nn.functional import pad
import os

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OutlierObserver): continue
        elif isinstance(m, OPTDecoderLayer):
            m.fc1 = W8A8Linear.from_float(m.fc1, weight_quant=weight_quant, act_quant=act_quant)
            m.fc2 = W8A8Linear.from_float(m.fc2, weight_quant=weight_quant, act_quant=act_quant)
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj = W8A8Linear.from_float(
                m.q_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.k_proj = W8A8Linear.from_float(
                m.k_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.v_proj = W8A8Linear.from_float(
                m.v_proj, weight_quant=weight_quant, act_quant=act_quant, quantize_output=quantize_bmm_input)
            m.out_proj = W8A8Linear.from_float(m.out_proj, weight_quant=weight_quant, act_quant=act_quant)
    return model


class EvaluatorTemp:
    def __init__(self, calib_dataset, valid_dataset, tokenizer, device):
        self.calib_dataset = calib_dataset
        self.valid_dataset = valid_dataset
        self.tokenizer = tokenizer
        self.device = device

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.calib_dataset = self.calib_dataset.map(tokenize_function, batched=True)
        self.valid_dataset = self.valid_dataset.map(tokenize_function, batched=True)

        self.calib_dataset.set_format(type='torch', columns=['input_ids'])
        self.valid_dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        for batch in self.valid_dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            label = input_ids[:, -1]
            outputs = model(input_ids)
            last_token_logits = outputs.logits[:, -2, :]
            pred = last_token_logits.argmax(dim=-1)
            total += label.size(0)
            hit += (pred == label).sum().item()
            break
        acc = hit / total
        return acc

    @torch.no_grad()
    def calibrate(self, model):
        model.eval()

        for idx, batch in enumerate(self.calib_dataset):
            print("[DEBUG] forward idx: ", idx)
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            model(input_ids)

class Evaluator:
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer

        # tokenize the dataset
        def tokenize_function(examples):
            example = self.tokenizer(examples['text'])
            return example

        self.dataset = self.dataset.map(tokenize_function, batched=True)
        self.dataset.set_format(type='torch', columns=['input_ids'])

    @torch.no_grad()
    def evaluate(self, model):
        model.eval()
        # The task is to predict the last word of the input.
        total, hit = 0, 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        latency = 0
        for batch in self.dataset:
            input_ids = batch['input_ids'].cuda().unsqueeze(0)
            label = input_ids[:, -1]
            pad_len = 512 - input_ids.shape[1]
            input_ids = pad(input_ids, (0, pad_len), value=1)
            torch.cuda.synchronize()
            start.record()
            outputs = model(input_ids)
            end.record()
            torch.cuda.synchronize()
            latency += start.elapsed_time(end)
            # last_token_logits = outputs.logits[:, -2-pad_len, :]
            # pred = last_token_logits.argmax(dim=-1)
            # total += label.size(0)
            # hit += (pred == label).sum().item()

        #acc = hit / total
        acc = 0
        lantecy = latency / len(self.dataset)
        return acc, lantecy


def set_outlier_axis(model):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1.in_observer.set_outlier_axis()
            m.fc2.in_observer.set_outlier_axis()
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj.in_observer.set_outlier_axis()
            m.k_proj.in_observer.set_outlier_axis()
            m.v_proj.in_observer.set_outlier_axis()
            m.out_proj.in_observer.set_outlier_axis()

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-13b')
    #train_dataset = load_dataset('lambada', split='train[:100]')
    calib_dataset = load_dataset('lambada', split='validation[:100]')
    valid_dataset = load_dataset('lambada', split='validation[100:1000]')
    #evaluator = Evaluator(train_dataset, valid_dataset, tokenizer, 'cuda')
    evaluator = Evaluator(calib_dataset, tokenizer)
    model_fp16 = OPTForCausalLM.from_pretrained('facebook/opt-13b', torch_dtype=torch.float16, device_map='auto')
    model_w8a8 = quantize_model(model_fp16)
    acc_w8a8 = evaluator.evaluate(model_w8a8)
    set_outlier_axis(model_w8a8)
    #evaluator.calibrate(model_w8a8)
    # For calibration dataset, calculate outlier axis and scale
    #calibrate_outlier(model_w8a8, train_dataset, 'cuda')

