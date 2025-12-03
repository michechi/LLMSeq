# Import Libraries ----------------------------------------------------------------
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import torch
import argparse
import logging
import os

from tqdm import tqdm
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, \
    precision_score, recall_score, classification_report, confusion_matrix
from scipy.stats import randint, uniform
from transformers import AutoTokenizer, AutoModel

# Setting Parsing & Logger ---------------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO, format="%(asctime)s: %(message)s")
logger = logging.getLogger(__name__)
parser = argparse.ArgumentParser()

def parse_args(args=None):

    parser.add_argument("--cache_dir", type=str, default="/cluster/work/projects/ec12/michechi/cache/",
                        help="Directory for cache e saving models")

    parser.add_argument("--tokenizer", choices=['LLM', 'Base', 'Countvect'])

    # Data
    parser.add_argument("--csv_to_use", type=str, help="CSV of training")

    parser.add_argument("--path_csv", type=str, default="/fp/homes01/u01/ec-michelec/MIMICIV/data/simulation/",
                        help="Path to file CSV for simulation data")
    
    # Seed
    parser.add_argument("--seed", type=int, default=9550,
                        help="Seed for riproducibility")
    
    # Where to save
    parser.add_argument("--output_dir", type=str, default="/fp/homes01/u01/ec-michelec/MIMICIV/src/output_slurms_fox/",
                        help="Directory to save outputs")
    
    # First partial parsing
    c_args, _ = parser.parse_known_args() # current_args

    # Adding contitioned arguments
    if c_args.tokenizer == "LLM":

        parser.add_argument("--model_name", required=True,
            help="Model name for tokenizer LLM")

        parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for DataLoader")

        parser.add_argument("--max_length", type=int, default=512,
                        help="Token Max length")
        
        parser.add_argument("--embedding_dir", type=str, default="/cluster/work/projects/ec12/michechi/embeddings/",
                        help="Directory for saving embeddings")
        
        parser.add_argument("--first_time", action="store_true",
                        help="First time of encoding embeddings?")

    # Final parse
    if args:
        args = parser.parse_args(args)
    else:
        args = parser.parse_args()

    # Log arguemnts values
    for arg, value in sorted(vars(args).items()):
        logger.info("Argument %s: %r", arg, value)

    return args

def ordinal_encoding(X_input):
    """
    Matrix (n_samples, 20) where each values it's 0-25
    """
    n_samples = len(X_input)
    n_cols = len(X_input.Sequences[0].split('\x1f'))
    X = np.zeros((n_samples, n_cols), dtype=int)
    
    for i, seq in enumerate(tqdm(X_input.Sequences)):
        for j, char in enumerate(seq.split('\x1f')):
            X[i, j] = ord(char.upper()) - ord('A')  # A=0, B=1, ..., Z=25
    
    df=pd.DataFrame(X)
    df=df.add_prefix("X_")
    return df

def standard_narrative_prompt(row, to_split='\x1f', column_name="Sequences"):
    events = row[column_name].split(to_split) # Testing on third-letter
    prompt = f'Sequential events: {" ".join(events)}\n'
    prompt += 'Outcome (0 or 1):'
    return prompt

def get_embeddings(texts,
                    model_name,
                    batch_size,
                    max_length,
                    device):
    
    logger.info(f"Loading {model_name} on {device}...")
    logger.info(f"Batch size: {batch_size}, Max length: {max_length}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name,
                                            token="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",
                                                cache_dir=args.cache_dir,
                                                force_download=True, 
                                                local_files_only=False)
    
    # Carica modello in FP16 se richiesto
    model = AutoModel.from_pretrained(
        model_name,
        token="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",
        torch_dtype=torch.bfloat16,
        device_map='auto',
        cache_dir=args.cache_dir,
        tie_word_embeddings=True
    ).to(device)
    model.eval()
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        
        inputs = tokenizer(
            batch_texts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=max_length
        ).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        # Converti a CPU e float32 per compatibilità con XGBoost
        all_embeddings.append(embeddings.cpu().float().numpy())
        
        # Libera memoria GPU
        del outputs, embeddings, inputs
        torch.cuda.empty_cache()
    
    return np.vstack(all_embeddings)


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Data Ingestion ----------------------------------------------------------------
    logger.info("Loading data...")
    X_train = pd.read_csv(f"{args.path_csv}X_train_{args.csv_to_use}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_train = pd.read_csv(f"{args.path_csv}y_train_{args.csv_to_use}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    X_val = pd.read_csv(f"{args.path_csv}X_val_{args.csv_to_use}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_val = pd.read_csv(f"{args.path_csv}y_val_{args.csv_to_use}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    X_test = pd.read_csv(f"{args.path_csv}X_test_{args.csv_to_use}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')
    y_test = pd.read_csv(f"{args.path_csv}y_test_{args.csv_to_use}.csv", na_values=['', 'None', 'NaN', 'na', 'nan']).fillna('')

    # Type of encoding
    if args.tokenizer == "Base":
        # Apply the base one
        X_train_encoded = ordinal_encoding(X_train)
        X_val_encoded = ordinal_encoding(X_val)
        X_test_encoded = ordinal_encoding(X_test)

    elif args.tokenizer == "LLM":
        safe_model_name = args.model_name.replace("/", "_")

        if args.first_time:
            # Apply tokenizer
            X_train['prompt'] = X_train.apply(standard_narrative_prompt, axis=1)
            X_val['prompt'] = X_val.apply(standard_narrative_prompt, axis=1)
            X_test['prompt'] = X_test.apply(standard_narrative_prompt, axis=1)
            
            # Train
            logger.info("\nProcessing train set...")
            X_train_encoded = get_embeddings(X_train['prompt'].tolist(), args.model_name, args.batch_size, args.max_length, device)
            np.save(f"{args.embedding_dir}X_train_{safe_model_name}_embeddings.npy", X_train_encoded)
            logger.info(f"Train embeddings shape: {X_train_encoded.shape}")

            # Val
            logger.info("\nProcessing validation set...")
            X_val_encoded = get_embeddings(X_val['prompt'].tolist(), args.model_name, args.batch_size, args.max_length, device)
            np.save(f"{args.embedding_dir}X_val_{safe_model_name}_embeddings.npy", X_val_encoded)
            logger.info(f"Val embeddings shape: {X_val_encoded.shape}")

            # Test
            logger.info("\nProcessing test set...")
            X_test_encoded = get_embeddings(X_test['prompt'].tolist(), args.model_name, args.batch_size, args.max_length, device)
            np.save(f"{args.embedding_dir}X_test_{safe_model_name}_embeddings.npy", X_test_encoded)
            logger.info(f"Test embeddings shape: {X_test_encoded.shape}")
        else:
            # If not first time, let's re-load embeddings
            X_train_encoded = np.load(f"{args.embedding_dir}X_train_{safe_model_name}_embeddings.npy")
            X_val_encoded = np.load(f"{args.embedding_dir}X_val_{safe_model_name}_embeddings.npy")
            X_test_encoded = np.load(f"{args.embedding_dir}X_test_{safe_model_name}_embeddings.npy")

    # Preparing data (merging train + val for CV)
    X_train_val = np.vstack([X_train_encoded, X_val_encoded])
    y_train_val = np.concatenate([y_train, y_val])


    # Definisci il modello base
    logger.info("Setting up XGBoost classifier and hyperparameter distributions...")
    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric='aucpr',
        tree_method='hist',
        device='cuda' 
    )

    # Distribuzioni per sampling
    logger.info("Defining hyperparameter distributions...")
    param_distributions = {
        'n_estimators': randint(50, 500),
        'max_depth': randint(3, 15),
        'learning_rate': uniform(0.01, 0.3),  # uniform tra 0.01 e 0.31
        'subsample': uniform(0.5, 0.5),  # uniform tra 0.5 e 1.0
        'colsample_bytree': uniform(0.5, 0.5),
        'min_child_weight': randint(1, 10),
        'gamma': uniform(0, 0.5)
    }
    logger.info(f"Parameter distributions: {param_distributions}")

    scoring = {
        'accuracy': 'accuracy',
        'precision': 'precision',
        'recall': 'recall',
        'f1': 'f1',
        'auc': 'roc_auc'
    }

    cpu_count = os.cpu_count()
    random_search = RandomizedSearchCV(
        estimator=xgb_base,
        param_distributions=param_distributions,
        n_iter=50,  # try 100 random combinations
        cv=5,
        scoring=scoring,
        refit='f1',
        n_jobs=1,
        verbose=2,
        random_state=args.seed
    )

    logger.info("Starting Randomized Search...")
    random_search.fit(X_train_val, y_train_val)

    logger.info(f"\nBest parameters: {random_search.best_params_}")
    logger.info(f"Best CV score: {random_search.best_score_:.4f}")

    # Converti risultati in DataFrame (molto "tidyverse"!)
    cv_results = pd.DataFrame(random_search.cv_results_)

    # Ordina per performance
    cv_results_sorted = cv_results.sort_values('rank_test_f1')

    # Top 10 configurazioni
    logger.info("\nTop 10 configurations:")
    logger.info(cv_results_sorted[['params', 'rank_test_f1', 'std_test_f1']].head(10))

    fig, ax = plt.subplots(figsize=(10, 6))

    # colormap (n linee = n profondità)
    depths = sorted(cv_results['param_max_depth'].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(depths)))
    markers = ['o', 's', 'D', '^', 'v', 'P', 'X']

    for i, depth in enumerate(depths):
        subset = cv_results[cv_results['param_max_depth'] == depth].sort_values('param_learning_rate')
        
        ax.plot(subset['param_learning_rate'], subset['mean_test_f1'],
                marker=markers[i % len(markers)],
                color=colors[i],
                linewidth=2,
                markersize=7,
                label=f"max_depth={depth}",
                alpha=0.9)

    ax.set_xlabel('Learning Rate', fontsize=12)
    ax.set_ylabel('Mean CV AUC', fontsize=12)
    ax.set_title('Hyperparameter Tuning Results', fontsize=14)

    ax.legend(title="max_depth", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Best model
    best_model = random_search.best_estimator_

    # Predizioni su test
    y_test_pred = best_model.predict(X_test_encoded)
    y_test_pred_proba = best_model.predict_proba(X_test_encoded)[:, 1]

    # Calcola tutte le metriche
    logger.info("\n=== TEST SET PERFORMANCE ===")
    logger.info(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    logger.info(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
    logger.info(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
    logger.info(f"F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
    logger.info(f"AUC:       {roc_auc_score(y_test, y_test_pred_proba):.4f}")

    logger.info("\n=== CLASSIFICATION REPORT ===")
    logger.info(classification_report(y_test, y_test_pred, target_names=['Not Ordered', 'Ordered']))

    logger.info("\n=== CONFUSION MATRIX ===")
    logger.info(confusion_matrix(y_test, y_test_pred))

    # # Salva modello
    # with open('best_xgb_model.pkl', 'wb') as f:
    #     pickle.dump(best_model, f)

    # # Salva anche i best params per reference
    # with open('best_params.txt', 'w') as f:
    #     f.write(str(random_search.best_params_))