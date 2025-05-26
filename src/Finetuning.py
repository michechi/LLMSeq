import os
import time
import logging
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, AdamW
)
from peft import LoraConfig, get_peft_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from huggingface_hub import login
import matplotlib.pyplot as plt

# Configura logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Universal finetuning script for LLMs or MedBERT-like models.")
    parser.add_argument("--model_type", type=str, choices=["llm", "medbert"], default="llm",
                        help="Tipo di modello: llm o medbert")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Nome del modello da Hugging Face")
    parser.add_argument("--peft", action="store_true", help="Usa PEFT (solo per LLM)")
    parser.add_argument("--cache_dir", type=str, default="/cluster/work/projects/ec403/ec-michechi/Project_M",
                        help="Directory per cache e salvataggio modelli")
    parser.add_argument("--input_csv", type=str, default="landmark_df.csv",
                        help="Path al file CSV di input")
    parser.add_argument("--max_visits", type=int, default=3,
                        help="Numero massimo di landmark da processare")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size per il DataLoader")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Numero di epoche di training")
    parser.add_argument("--patience", type=int, default=2,
                        help="Early stopping patience")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Lunghezza massima dei token")
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Seed per riproducibilità")
    return parser.parse_args()

def load_tokenizer(model_name, model_type, hf_token, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if model_type == "llm" else None,
        cache_dir=cache_dir
    )
    
    return tokenizer

def load_model(model_name, model_type, tokenizer, cache_dir, hf_token, use_peft):
    if model_type == "llm":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            token=hf_token,
            cache_dir=cache_dir
        )
        model.config.pad_token_id = tokenizer.eos_token_id
        if use_peft:
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0.1,
                bias='none',
                task_type="SEQ_CLS"
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
    else:  # e.g., MedBERT, ClinicalBERT
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            cache_dir=cache_dir
        )
    return model

class ClinicalDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=256):
        self.encodings = tokenizer(
            texts, truncation=True, padding=True, max_length=max_length
        )
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def narrative_prompt(row):
    narrative = f"Patient is a {row['age_at_landmark']}-year-old {row['gender']}."
    narrative += f" This is the {row['num_total_visits']} visit."
    if row['days_since_previous_visit'] != -1:
        narrative += f" The last visit happened {row['days_since_previous_visit']} days ago."
    if pd.notna(row['diag_text']) and row['diag_text'].strip():
        narrative += f" Medical history includes: {row['diag_text']}."
    if pd.notna(row['med_text']) and row['med_text'].strip():
        narrative += f" Current medications are: {row['med_text']}."
    if pd.notna(row['proc_text']) and row['proc_text'].strip():
        narrative += f" Procedures performed: {row['proc_text']}."
    return narrative

def train_and_evaluate(model, train_loader, val_loader, args, landmark_visit):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )
    best_auc = 0.0
    epochs_no_improve = 0
    best_model_path = get_best_model_path(args.model_name, landmark_visit, args.cache_dir)

    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        val_preds, val_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch['labels'].cpu().numpy())
        val_auc = roc_auc_score(val_labels, val_preds)
        logger.info(
            f"Landmark {landmark_visit} | Epoch {epoch+1}/{args.epochs} | "
            f"Train Loss: {avg_train_loss:.4f} | Validation AUC: {val_auc:.4f}"
        )

        if val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
    model.load_state_dict(torch.load(best_model_path))
    return best_auc, best_model_path, val_preds, val_labels

def get_best_model_path(model_name, landmark_visit, cache_dir):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace("/", "_")
    filename = f"best_model_{safe_model_name}_landmark{landmark_visit}_{timestamp}.pt"
    best_model_dir = os.path.join(cache_dir, 'best')
    os.makedirs(best_model_dir, exist_ok=True)
    return os.path.join(best_model_dir, filename)

def bootstrap_auc_ci(y_true, y_pred, n_bootstraps=1000, alpha=0.95, seed=42):
    bootstrapped_scores = []
    rng = np.random.RandomState(seed)
    for _ in range(n_bootstraps):
        indices = rng.randint(0, len(y_pred), len(y_pred))
        if len(np.unique(np.array(y_true)[indices])) < 2:
            continue
        score = roc_auc_score(np.array(y_true)[indices], np.array(y_pred)[indices])
        bootstrapped_scores.append(score)
    sorted_scores = np.array(bootstrapped_scores)
    sorted_scores.sort()
    lower = sorted_scores[int((1.0 - alpha) / 2 * len(sorted_scores))]
    upper = sorted_scores[int((alpha + (1.0 - alpha) / 2) * len(sorted_scores))]
    return lower, upper

def main():
    args = parse_args()
    hf_token = os.getenv("HF_TOKEN")
    if args.model_type == "llm" and hf_token is None:
        raise ValueError("Set the HF_TOKEN environment variable for authentication.")
    if args.model_type == "llm":
        login(hf_token)

    tokenizer = load_tokenizer(args.model_name, args.model_type, hf_token, args.cache_dir)
    model = load_model(args.model_name, args.model_type, tokenizer, args.cache_dir, hf_token, args.peft)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer non ha un pad_token. Lo aggiungo manualmente come [PAD].")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))

    landmark_df = pd.read_csv(args.input_csv)
    visit_counts = landmark_df['subject_id'].value_counts()
    selected_patients = visit_counts[visit_counts == args.max_visits].index
    df_selected = landmark_df[landmark_df['subject_id'].isin(selected_patients)].copy()

    predictions_list, labels_list, results = [], [], []

    for landmark_visit in range(1, args.max_visits + 1):
        logger.info(f"Fine-tuning at Landmark {landmark_visit}")
        
        start_time = datetime.datetime.now()
        
        df_subset = df_selected[df_selected['landmark_visit'] == landmark_visit]
        patients = df_subset['subject_id'].unique()
        train_patients, test_patients = train_test_split(
            patients, test_size=0.2, random_state=args.seed,
            stratify=df_subset.groupby('subject_id')['death_in_90days'].max()
        )
        train_df = df_subset[df_subset['subject_id'].isin(train_patients)].copy()
        test_df = df_subset[df_subset['subject_id'].isin(test_patients)].copy()
        train_texts = train_df.apply(narrative_prompt, axis=1).tolist()
        train_labels = train_df['death_in_90days'].tolist()
        test_texts = test_df.apply(narrative_prompt, axis=1).tolist()
        test_labels = test_df['death_in_90days'].tolist()

        train_dataset = ClinicalDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
        test_dataset = ClinicalDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

        best_auc, best_model_path, preds, true_labels = train_and_evaluate(
            model, train_loader, test_loader, args, landmark_visit
        )
        elapsed_time = datetime.datetime.now() - start_time
        elapsed_seconds = elapsed_time.total_seconds()
        logger.info(f"Tempo impiegato per Landmark {landmark_visit}: {elapsed_seconds:.1f} secondi")

        predictions_list.append(preds)
        labels_list.append(true_labels)
        results.append({
            'Landmark': landmark_visit,
            'AUC': f"{best_auc:.4f}",
            'Patients': len(test_df),
            'Model Path': best_model_path,
            'Training Time (s)': f"{elapsed_seconds:.1f}"
        })

    # Confidence intervals + plot
    auc_means, ci_lowers, ci_uppers = [], [], []
    for preds, labels in zip(predictions_list, labels_list):
        auc = roc_auc_score(labels, preds)
        ci_lower, ci_upper = bootstrap_auc_ci(labels, preds, seed=args.seed)
        auc_means.append(auc)
        ci_lowers.append(ci_lower)
        ci_uppers.append(ci_upper)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, args.max_visits + 1), auc_means, marker='o', label='Mean AUC')
    plt.fill_between(range(1, args.max_visits + 1), ci_lowers, ci_uppers, alpha=0.2, label='95% CI')
    plt.xlabel('Landmark Visit')
    plt.ylabel('AUC')
    plt.title(f'{args.model_name} Predictive Performance')
    plt.legend()
    plt.grid(alpha=0.4)
    plt.ylim([0.5, 1.0])
    plt.show()

    results_df = pd.DataFrame(results)
    logger.info(results_df)

if __name__ == "__main__":
    main()