import torch
from transformers.models.opt.modeling_opt import OPTAttention, OPTDecoderLayer, OPTForCausalLM
from transformers import GPT2Tokenizer
from smoothquant.smooth import smooth_lm
from smoothquant.fake_quant import W8A8Linear
from datasets import load_dataset

def quantize_model(model, weight_quant='per_tensor', act_quant='per_tensor', quantize_bmm_input=True):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
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


class Evaluator:
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

        for batch in self.calib_dataset:
            input_ids = batch['input_ids'].to(self.device).unsqueeze(0)
            model(input_ids)

def set_outlier_axis(model):
    for name, m in model.model.named_modules():
        if isinstance(m, OPTDecoderLayer):
            m.fc1.set_outlier_axis()
            m.fc2.set_outlier_axis()
        elif isinstance(m, OPTAttention):
            # Her we simulate quantizing BMM inputs by quantizing the output of q_proj, k_proj, v_proj
            m.q_proj.set_outlier_axis()
            m.k_proj.set_outlier_axis()
            m.v_proj.set_outlier_axis()
            m.out_proj.set_outlier_axis()

if __name__ == '__main__':
    tokenizer = GPT2Tokenizer.from_pretrained('facebook/opt-13b')
    train_dataset = load_dataset('lambada', split='train[:100]')
    valid_dataset = load_dataset('lambada', split='validation[:1000]')
    evaluator = Evaluator(train_dataset, valid_dataset, tokenizer, 'cuda')
    model_fp16 = OPTForCausalLM.from_pretrained('facebook/opt-13b', torch_dtype=torch.float16, device_map='cuda')
    model_w8a8 = quantize_model(model_fp16, act_quant='per_token')

    evaluator.calibrate(model_w8a8)
    # For calibration dataset, calculate outlier axis and scale
    #calibrate_outlier(model_w8a8, train_dataset, 'cuda')

