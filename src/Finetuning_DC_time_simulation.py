import os
import gc
import ast
import random
import logging
import datetime
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup, BitsAndBytesConfig, AutoModelForCausalLM
)
from peft import LoraConfig, get_peft_model
from sklearn.metrics import roc_auc_score, f1_score
from huggingface_hub import login

torch.set_float32_matmul_precision('high') 
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)

def standard_narrative_prompt(row, to_split='\x1f'):
    events = row.split(to_split)
    prompt = f'Sequential events: {" ".join(events)}\n'
    prompt += 'Outcome (0 or 1):'
    return prompt

def set_seed(seed_value=5550):
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    # torch.use_deterministic_algorithms(True)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def parse_args():

    parser = argparse.ArgumentParser(description="Testing LLMs on Generated Dataset")

    # Model-General Settings
    parser.add_argument("--model_type", type=str, choices=["general", "medical"], default="general",
                        help="Model type: general-purpose o medical-purpose (MedBERT-like)")

    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the model from Hugging Face")

    parser.add_argument("--peft", action="store_true", help="Usa PEFT (solo per LLM)")

    parser.add_argument("--use_quantization", action="store_true",
                        help="Quantization 4 bit")

    parser.add_argument("--cache_dir", type=str, default="/root/MIMICIV/cache",
                        help="Directory for cache e saving models")

    # Data
    ## Train
    parser.add_argument("--X_train_csv", type=str, default="/root/MIMICIV/data/simulation/X_train.csv",
                        help="Path to file CSV di training")
    
    parser.add_argument("--y_train_csv", type=str, default="/root/MIMICIV/data/simulation/y_train.csv",
                        help="Path to file CSV di training")
    ## Val
    parser.add_argument("--X_val_csv", type=str, default="/root/MIMICIV/data/simulation/X_val.csv",
                        help="Path to file CSV di validation")
    
    parser.add_argument("--y_val_csv", type=str, default="/root/MIMICIV/data/simulation/y_val.csv",
                        help="Path to file CSV di training")
    ## Test
    parser.add_argument("--X_test_csv", type=str, default="/root/MIMICIV/data/simulation/X_test.csv",
                        help="Path to file CSV di validation")
    
    parser.add_argument("--y_test_csv", type=str, default="/root/MIMICIV/data/simulation/y_test.csv",
                        help="Path to file CSV di training")

    # Prompts
    parser.add_argument("--prompt_type", type=str, choices=['standard'], default="standard",
                        help="Prompting type to use.")

    # Hyperparams
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for DataLoader")

    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Number of steps for gradient accumulation (useful for large models)")

    parser.add_argument("--epochs", type=int, default=20,
                        help="Num epochs for training")

    parser.add_argument("--patience", type=int, default=3,
                        help="Early stopping patience")

    parser.add_argument("--max_length", type=int, default=512,
                        help="Token Max lenght")

    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Learning rate")

    parser.add_argument("--early", type=str, choices=["auc", "loss", "f1"], default="loss",
                        help="Early stopping criterion: 'auc', 'loss' or 'f1")

    parser.add_argument("--seed", type=int, default=9550,
                        help="Seed for riproducibility")
    
    args = parser.parse_args()

    # Log arguemnts values
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args

# New class - CausalLM
class CausalLMWithClassificationHead(nn.Module):
    """
    Wrapper che combina CausalLM con testa di classificazione
    
    Il modello impara CONTEMPORANEAMENTE:
    1. Next token prediction (per imparare pattern temporali)
    2. Mortality classification (per il task finale)
    """
    
    def __init__(self, backbone_model, num_classes=2):
        super().__init__()
        
        self.backbone = backbone_model
        self.config = backbone_model.config
        self.num_classes = num_classes
        
        # Testa di classificazione che opera sugli hidden states
        self.classification_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size // 2),
            nn.Tanh(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size // 2, num_classes)
        )

        self.classification_head = self.classification_head.to(
            dtype=backbone_model.dtype, 
            device=backbone_model.device
            )
        
        
    def forward(self, input_ids, attention_mask=None, labels=None, outcome_labels=None):
        """
        Forward pass che calcola ENTRAMBE le loss:
        1. Causal LM loss (next token prediction)
        2. Classification loss (mortality prediction)
        """
        
        # 1. Pass attraverso il backbone CausalLM
        causal_outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,  # Per next token prediction
            output_hidden_states=True,
            return_dict=True
        )
        
        # 2. Estrai hidden states per la classificazione
        hidden_states = causal_outputs.hidden_states[-1]  # Ultimo layer
        
        if outcome_labels is not None:
            # Estrai rappresentazioni alle posizioni del token mortality
            batch_size = input_ids.shape[0]
            classification_representations = []
            
            for b in range(batch_size):
                if attention_mask is not None:
                    # Usa attention mask
                    last_token_pos = attention_mask[b].sum().item() - 1
                else:
                    # Trova l'ultimo token non-padding manualmente
                    sequence = input_ids[b]
                    pad_token_id = self.backbone.config.pad_token_id
                    
                    # Trova l'ultima posizione che non è padding
                    last_token_pos = len(sequence) - 1
                    while last_token_pos >= 0 and sequence[last_token_pos] == pad_token_id:
                        last_token_pos -= 1
                    
                    # Failsafe: se tutto è padding, usa posizione 0
                    last_token_pos = max(0, last_token_pos)
                    
                # Estrai rappresentazione per classificazione
                classification_representations.append(hidden_states[b, last_token_pos, :])
            
            # Stack in un tensor
            classification_input = torch.stack(classification_representations)
            
            # 4. Calcola logits di classificazione
            classification_logits = self.classification_head(classification_input)
            
            # 5. Calcola classification loss
            classification_loss = nn.functional.cross_entropy(
                classification_logits, outcome_labels
            )
            
            # 6. Combina le loss
            total_loss = causal_outputs.loss + classification_loss
            
            return {
                'loss': total_loss,
                'causal_loss': causal_outputs.loss,
                'classification_loss': classification_loss,
                'logits': classification_logits,  
                'causal_logits': causal_outputs.logits,
                'hidden_states': hidden_states
            }
        
        else:
            # Solo inference - estrai rappresentazioni per classificazione
            batch_size = input_ids.shape[0]
            classification_representations = []
            
            for b in range(batch_size):
                if attention_mask is not None:
                    last_token_pos = attention_mask[b].sum().item() - 1
                else:
                    last_token_pos = input_ids.shape[1] - 1
                    
                classification_representations.append(hidden_states[b, last_token_pos, :])
            
            classification_input = torch.stack(classification_representations)
            classification_logits = self.classification_head(classification_input)
            
            return {
                'loss': causal_outputs.loss,
                'logits': classification_logits,
                'causal_logits': causal_outputs.logits,
                'hidden_states': hidden_states
            }

    def resize_token_embeddings(self, new_num_tokens):
        """Delega al modello backbone"""
        return self.backbone.resize_token_embeddings(new_num_tokens)


def load_tokenizer(model_name, model_type, hf_token, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token if model_type == "general" else None,
        cache_dir=cache_dir,
        force_download=True,
        local_files_only=False
    )
    return tokenizer

def load_model_causal(model_name, model_type, tokenizer, cache_dir, hf_token, use_peft, use_quantization):
    """
    Versione modificata che carica CausalLM invece di SequenceClassification
    """
    if model_type == "general":
        if use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None

        # CARICA CAUSAL LM INVECE DI SEQUENCE CLASSIFICATION
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map='auto',
            token=hf_token,
            cache_dir=cache_dir,
            quantization_config=bnb_config if use_quantization else None
        )
        
        base_model.config.pad_token_id = tokenizer.pad_token_id
        
        # WRAPPER CON TESTA DI CLASSIFICAZIONE
        model = CausalLMWithClassificationHead(base_model, num_classes=2)
        
        if use_peft:
            # PEFT solo sul backbone, non sulla testa
            lora_config = LoraConfig(
                r=8,
                lora_alpha=16,
                target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
                lora_dropout=0.1,
                bias='none',
                task_type="CAUSAL_LM"  # Cambiato da SEQ_CLS
            )
            model.backbone = get_peft_model(model.backbone, lora_config)
            model.backbone.print_trainable_parameters()
    else:
        # Per modelli medici, mantieni il comportamento originale
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2,
            cache_dir=cache_dir
        )
    
    return model

class TemporalCausalDataset:
    """
    Dataset che prepara sequenze per apprendimento causale + classificazione
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
            
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        
        # Tokenizza la sequenza completa
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        
        # Per causal LM: labels = input_ids shifted
        causal_labels = input_ids.clone()

        return {
            'input_ids': input_ids,
            'labels': causal_labels,  # Per next token prediction
            'mortality_labels': torch.tensor(label, dtype=torch.long)  # Per classificazione
        }

def train_and_evaluate_causal(model, train_loader, val_loader, args):
    """
    Training loop modificato per gestire loss combinate
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model = model.to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    total_steps = args.epochs * len(train_loader)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.06 * total_steps),
        num_training_steps=total_steps
    )
    
    best_auc, best_f1, best_val_loss = 0.0, 0.0, float("inf")
    epochs_no_improve = 0
    best_model_path = get_best_model_path(args)
    
    logger.info(f"Early stopping criterion: {args.early}")
    
    for epoch in range(args.epochs):
        model.train()
        total_train_loss = 0
        total_causal_loss = 0
        total_classification_loss = 0
        
        optimizer.zero_grad()
        
        for step, batch in enumerate(train_loader):
            inputs = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**inputs)
            
            # Loss totale (causal + classification)
            loss = outputs['loss'] / args.gradient_accumulation_steps
            causal_loss = outputs['causal_loss'] / args.gradient_accumulation_steps
            classification_loss = outputs['classification_loss'] / args.gradient_accumulation_steps
            loss.backward()
            
            total_train_loss += loss.item()
            total_causal_loss += causal_loss.item()
            total_classification_loss += classification_loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_causal_loss = total_causal_loss / len(train_loader)
        avg_classification_loss = total_classification_loss / len(train_loader)

        # VALIDATION
        model.eval()
        total_val_loss = 0
        val_preds, val_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**inputs)
                
                val_loss = outputs['loss']
                total_val_loss += val_loss.item()
                
                # Usa i logits di classificazione (non quelli causali)
                probs = torch.softmax(outputs['logits'], dim=-1)[:, 1].cpu().float().numpy()
                val_preds.extend(probs)
                val_labels.extend(batch['mortality_labels'].cpu().numpy())
                
        avg_val_loss = total_val_loss / len(val_loader)
        val_auc = roc_auc_score(val_labels, val_preds)
        
        threshold = 0.5
        val_preds_binary = (np.array(val_preds) >= threshold).astype(int)
        val_f1 = f1_score(val_labels, val_preds_binary, zero_division=0)

        logger.info(
            f"Epoch {epoch+1}/{args.epochs} | "
            f"Total Loss: {avg_train_loss:.4f} | "
            f"Causal Loss: {avg_causal_loss:.4f} | "
            f"Classification Loss: {avg_classification_loss:.4f} | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Val AUC: {val_auc:.4f} | "
            f"F1-Score: {val_f1:.4f}"
        )
        
        # Early stopping (stesso del tuo codice)
        condition = False
        if args.early == "auc":
            condition = val_auc > best_auc
        elif args.early == "loss":
            condition = avg_val_loss < best_val_loss
        elif args.early == "f1":
            condition = val_f1 > best_f1

        if condition:
            best_auc = val_auc
            best_f1 = val_f1
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"New best model saved with AUC: {best_auc:.4f}")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
    if args.use_quantization:
        model.load_state_dict(torch.load(best_model_path), strict=False)
    else:
        model.load_state_dict(torch.load(best_model_path), strict=True)
    return best_auc, best_model_path, best_f1, best_val_loss, epoch + 1

def get_best_model_path(args):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = args.model_name.replace("/", "_")
    filename = f"best_model_{safe_model_name}_{args.seed}_{args.prompt_type}_{args.max_visits}_{args.max_length}_{args.when_counting_death}_{timestamp}_.pt"
    best_model_dir = os.path.join(args.cache_dir, 'best')
    os.makedirs(best_model_dir, exist_ok=True)
    return os.path.join(best_model_dir, filename)

def main():
    args = parse_args()

    # Set seed for reproducibility
    set_seed(args.seed)

    #hf_token = os.getenv("HF_TOKEN")
    hf_token = "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"

    tokenizer = load_tokenizer(args.model_name, args.model_type, hf_token, args.cache_dir)
    model = load_model_causal(args.model_name, args.model_type, tokenizer, args.cache_dir, hf_token, args.peft, args.use_quantization).to(device="cpu")
    
    if tokenizer.pad_token is None:
        logger.warning("Tokenizer non ha un pad_token. Lo aggiungo manualmente come [PAD].")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    # After having resized the model, move it to the appropriate device    
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    if args.prompt_type == "standard":
        narrative_prompt = standard_narrative_prompt
    else:
        raise ValueError("Invalid prompt type. Use 'naive' or 'compact'.")
    logger.info(f"Using prompt type: {args.prompt_type}")
    
    logger.info(f"Model type: {args.model_type}, Model name: {args.model_name}") 

    # Reading data - already splitted!
    X_train = pd.read_csv(args.X_train_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_train = pd.read_csv(args.y_train_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    
    X_val = pd.read_csv(args.X_val_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_val = pd.read_csv(args.y_val_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    X_test = pd.read_csv(args.X_test_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_test = pd.read_csv(args.y_test_csv, na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    
    results = []
    
    # Set the seed for reproducibility
    set_seed(args.seed)
    
    gc.collect()
    torch.cuda.empty_cache()

    # Load model and tokenizer
    hf_token = "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"
    if args.model_type == "general" and hf_token is None:
        raise ValueError("Set the HF_TOKEN environment variable for authentication.")
    if args.model_type == "general":
        login(hf_token)

    tokenizer = load_tokenizer(args.model_name, args.model_type, hf_token, args.cache_dir)    
    model = load_model_causal(args.model_name, args.model_type, tokenizer, args.cache_dir, hf_token, args.peft, args.use_quantization).to(device="cpu")

    if tokenizer.pad_token is None:
        logger.warning("Tokenizer non ha un pad_token. Lo aggiungo manualmente come [PAD].")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        model.resize_token_embeddings(len(tokenizer))
        
    # After having resized the model, move it to the appropriate device    
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    train_texts = X_train.apply(narrative_prompt, axis=1).tolist()
    val_texts = X_val.apply(narrative_prompt, axis=1).tolist()
    test_texts = X_test.apply(narrative_prompt, axis=1).tolist()
    
    logger.info(f"Example train text: {train_texts[0]}")

    train_labels = y_train["Outcome"].tolist()
    val_labels = y_val["Outcome"].tolist()
    test_labels = y_test["Outcome"].tolist()

    # Media di lunghezza dei testi
    avg_train_length = np.mean([len(text.split()) for text in train_texts])
    avg_val_length = np.mean([len(text.split()) for text in val_texts])
    avg_test_length = np.mean([len(text.split()) for text in test_texts])
    logger.info(f"Average train text length: {avg_train_length:.2f} words")
    logger.info(f"Average validation text length: {avg_val_length:.2f} words")
    logger.info(f"Average test text length: {avg_test_length:.2f} words")
    logger.info(f"Training on {len(train_texts)} samples, validating on {len(val_texts)}, testing on {len(test_texts)} samples")

    train_dataset = TemporalCausalDataset(train_texts, train_labels, tokenizer, max_length=args.max_length)
    val_dataset = TemporalCausalDataset(val_texts, val_labels, tokenizer, max_length=args.max_length)
    test_dataset = TemporalCausalDataset(test_texts, test_labels, tokenizer, max_length=args.max_length)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    start_time = datetime.datetime.now()
    best_auc, best_model_path, best_val_f1, best_val_loss, epochs_done = train_and_evaluate_causal(
        model, train_loader, val_loader, args
    )
    elapsed_time = datetime.datetime.now() - start_time
    elapsed_seconds = elapsed_time.total_seconds()
    logger.info(f"Tempo medio per epoca: {elapsed_seconds / epochs_done:.1f} secondi") # This is the important one

    # VALUTAZIONE FINALE SUL TEST SET
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    logger.info(f"Loading best model from {best_model_path} for final evaluation on test set")
    if args.use_quantization:
        model.load_state_dict(torch.load(best_model_path, map_location=device), strict=False)
    else:
        model.load_state_dict(torch.load(best_model_path, map_location=device), strict=True)
    model.eval()
    test_preds, test_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            inputs = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)[:, 1].cpu().float().numpy() # on cpu for sklearn metrics
            test_preds.extend(probs)
            test_labels.extend(batch['mortality_labels'].cpu().float().numpy()) # on cpu for sklearn metrics

    test_auc = roc_auc_score(test_labels, test_preds)
    test_f1 = f1_score(test_labels, np.array(test_preds) >= 0.5)

    logger.info(
        f"Final Test AUC: {test_auc:.4f} | "
        f"Test F1-Score: {test_f1:.4f}"
    )

    results.append({
        'Val AUC': f"{best_auc:.4f}",
        'Val F1-Score': f"{best_val_f1:.4f}",
        'Test AUC': f"{test_auc:.4f}",
        'Test F1-Score': f"{test_f1:.4f}",
        'Patients (Test)': len(X_test),
        'Model Path': best_model_path,
        'Validation Loss': f"{best_val_loss:.4f}",
        'Training Time (s)': f"{elapsed_seconds:.1f}"
    })
   
    results_df = pd.DataFrame(results)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model_name.replace("/", "_")
    peft_tag = "_peft" if args.peft else ""
    output_filename = (
        f"results_{args.model_type}_{model_tag}_{args.when_counting_death}_visits{args.max_visits}{peft_tag}_{args.seed}_{args.max_length}_{args.prompt_type}_{timestamp}.csv"
    )
    logger.info(f"Saving results to {output_filename}")
    file_dir = f"{args.cache_dir}/results/{args.model_type}/{args.when_counting_death}/visits{args.max_visits}" 
    os.makedirs(file_dir, exist_ok=True)
    results_df.to_csv(os.path.join(file_dir, output_filename), index=False)

    print(results_df)

if __name__ == "__main__":
    main()