import argparse
import torch
import os
import json
import ipdb
import math
from tqdm import tqdm
import shortuuid
import whisper

from omni_speech.constants import SPEECH_TOKEN_INDEX, DEFAULT_SPEECH_TOKEN
from omni_speech.conversation import conv_templates
from omni_speech.model.builder import load_pretrained_model,create_model
from omni_speech.model.audio_process.wav import preprocess_audio
from omni_speech.utils import disable_torch_init
from omni_speech.datasets.preprocess import tokenizer_speech_token
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from transformers import DataCollatorForLanguageModeling
from transformers import DataCollactorForSeq2Seq
from transformers import TrainingArguments
from transformers import Trainer


def collate_fn(batch):
    for i in range(len(batch)):
        batch[i] = batch[i].values()

    input_ids,labels,speech_tensors, speech_lengths = zip(*batch)
    input_ids = pad_sequence(input_ids,batch_first=True,padding_value=128009)
    labels = pad_sequence(labels,batch_first=True,padding_value=128009)

    speech_tensors = torch.stack(speech_tensors, dim=0)
    speech_lengths = torch.stack(speech_lengths, dim=0)
    return {"input_ids":input_ids,"labels":labels,"speech":speech_tensors,"speech_lengths":speech_lengths}


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, questions, tokenizer, model_config, input_type, mel_size, conv_mode):
        self.questions = questions
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size

    def __getitem__(self, index):
        item = self.questions[index]
        speech_file = item["speech"]
        qs = item["conversations"][0]["value"]
        re = item["conversations"][1]["value"]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], re)
        prompt = conv.get_prompt()

        #speech_file是音频文件
        #speech = preprocess_audio(speech_file, target_sample_rate=16000, n_mels=self.mel_size, sequence_length=128)

        speech = whisper.load_audio(speech_file)
        if self.input_type == "raw":
            speech = torch.from_numpy(speech)
            if self.model_config.speech_normalize:
                speech = torch.nn.functional.layer_norm(speech, speech.shape)
        elif self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors='pt')
        ret = dict(input_ids=input_ids,labels=input_ids,speech=speech.to(torch.bfloat16),speech_lengths=torch.LongTensor([speech.shape[0]]))
        return ret

    def __len__(self):
        return len(self.questions)

# DataLoader
def create_data_loader(questions, tokenizer, model_config, input_type, mel_size, conv_mode, batch_size=2, num_workers=4):
    #assert batch_size == 1, "batch_size must be 1"
    dataset = CustomDataset(questions, tokenizer, model_config, input_type, mel_size)
    #data_loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, collate_fn=collate_fn)
    #为啥这里用data_loader，而不是直接用dataset？
    return dataset

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks"""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]


def train_model(args):
    # Model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = os.path.expanduser(args.model_path)
    tokenizer, model, context_len = create_model(model_path, args.model_base, is_lora=args.is_lora, s2s=args.s2s)

    questions = json.load(open(os.path.expanduser(args.question_file), "r"))
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    data_loader = create_data_loader(questions, tokenizer, model.config, args.input_type, args.mel_size)

    for (input_ids, speech_tensor, speech_length), item in tqdm(zip(data_loader, questions), total=len(questions)):
        idx = item["id"]
        try:
            answer = item["conversations"][1]["value"]
        except:
            answer = None
        input_ids = input_ids.to(device='cuda', non_blocking=True)
        speech_tensor = speech_tensor.to(dtype=torch.float16, device='cuda', non_blocking=True)
        speech_length = speech_length.to(device='cuda', non_blocking=True)

        with torch.inference_mode():
            if args.s2s:
                outputs = model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                    streaming_unit_gen=False,
                )
                output_ids, output_units = outputs
            else:
                outputs = model.generate(
                    input_ids,
                    speech=speech_tensor,
                    speech_lengths=speech_length,
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    max_new_tokens=args.max_new_tokens,
                    use_cache=True,
                    pad_token_id=128004,
                )
                output_ids = outputs

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        if args.s2s:
            output_units = ctc_postprocess(output_units, blank=model.config.unit_vocab_size)

        print(f"H-{idx}\t{outputs}")
        print(f"T-{idx}\t{answer}")
        if args.s2s:
            print(f"U-{idx}\t{output_units}")

        if args.s2s:
            ans_file.write(json.dumps({"question_id": idx, "prediction": outputs, "prediction_units": output_units, "answer": answer}) + "\n")
        else:
            ans_file.write(json.dumps({"question_id": idx, "prediction": outputs, "answer": answer}) + "\n")
        # ans_file.flush()
    ans_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--question-file", type=str)
    parser.add_argument("--answer-file", type=str)
    parser.add_argument("--conv-mode", type=str, default="v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--input_type", type=str, default="raw")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--s2s", action="store_true", default=False)
    parser.add_argument("--is_lora", action="store_true", default=False)
    args = parser.parse_args()

    eval_model(args)
