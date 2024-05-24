from transformers import AutoProcessor
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
from transformers import AutoModelForPreTraining
from peft import get_peft_model, LoraConfig, TaskType
from transformers import AdamW


batch_size = 2
accumulation_steps = 2
num_epochs = 3

dataset_name = "waleko/TikZ-short-code"

raw_datasets = load_dataset(dataset_name)
train_dataset = raw_datasets["train"]
if "test" not in raw_datasets.keys():
    print("Splitting into train and test...")
    ds = train_dataset.train_test_split(0.2, shuffle=True, seed=42)
    train_dataset, eval_dataset = ds["train"], ds["test"]
else:
    eval_dataset = raw_datasets["test"]
model_name = "llava-hf/llava-1.5-7b-hf"

processor = AutoProcessor.from_pretrained(model_name)


def collate_fn(examples):
    texts = []
    images = []
    for example in examples:
        code = example["code"]
        text = (
            "Assistant helps to write down the TikZ code for the user's image. USER: <image>\nWrite down the TikZ code to draw the diagram shown in the image. ASSISTANT: "
            + code
            + "</s>"
        )
        texts.append(text)
        images.append(example["image"])
    batch = processor(
        texts,
        images,
        return_tensors="pt",
        padding=True,
        max_length=512,
        truncation=True,
    )

    labels = batch["input_ids"].clone()
    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100
    batch["labels"] = labels

    return batch


train_dataloader = DataLoader(
    train_dataset, shuffle=True, batch_size=batch_size, collate_fn=collate_fn
)
eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)

orig_model = AutoModelForPreTraining.from_pretrained(model_name).cuda()

peft_config = LoraConfig(
    r=64,
    target_modules="all-linear",
    task_type=TaskType.CAUSAL_LM,
    lora_alpha=6,
)
model = get_peft_model(orig_model, peft_config)
model.print_trainable_parameters()
num_training_steps = num_epochs * len(train_dataloader)
device = "cuda"


optimizer = AdamW(model.parameters(), lr=5e-4)

from transformers import get_linear_schedule_with_warmup

scheduler = get_linear_schedule_with_warmup(
    optimizer, int(num_training_steps * 0.03), num_training_steps
)

import wandb

wandb.init(project="tikz-llava")

from tqdm.auto import tqdm

progress_bar = tqdm(range(num_training_steps))

for epoch in range(num_epochs):
    model.train()
    for batch_idx, batch in enumerate(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss = loss / accumulation_steps  # Normalize the loss

        wandb.log(
            {"train/loss": loss.item() * accumulation_steps}
        )  # Log the original loss
        loss.backward()

        # Perform optimizer step and reset gradients every accumulation_steps

        if ((batch_idx + 1) % accumulation_steps == 0) or (
            batch_idx + 1 == len(train_dataloader)
        ):
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
        progress_bar.update(1)
    model.eval()
    eval_loss = 0
    eval_steps = 0
    for batch in eval_dataloader:
        with torch.no_grad():
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.item()
            eval_steps += 1
    avg_eval_loss = eval_loss / eval_steps
    wandb.log({"eval/loss": avg_eval_loss})
    print(f"Epoch {epoch} | Eval Loss: {avg_eval_loss}")
model.save_pretrained("checkpoint-peft", save_embedding_layers=True)

wandb.finish()
