import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import Optional, Tuple, List
import numpy as np

class TemporalClassificationHead(nn.Module):
    """
    Testa di classificazione che incorpora informazioni temporali
    """
    def __init__(self, hidden_size: int, num_visits: int, num_classes: int = 2):
        super().__init__()
        
        # Parametri
        self.hidden_size = hidden_size
        self.num_visits = num_visits
        self.num_classes = num_classes
        
        # Layer per elaborare le rappresentazioni di ogni visita
        self.visit_encoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size // 2)
        )
        
        # LSTM per catturare dipendenze temporali tra visite
        self.temporal_lstm = nn.LSTM(
            input_size=hidden_size // 2,
            hidden_size=hidden_size // 4,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
            bidirectional=True
        )
        
        # Attention mechanism per pesare l'importanza delle visite
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size // 2,  # bidirectional LSTM output
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classificatore finale
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 4, num_classes)
        )
    
    def forward(self, visit_representations: torch.Tensor, 
                visit_mask: Optional[torch.Tensor] = None):
        """
        Args:
            visit_representations: (batch_size, num_visits, hidden_size)
            visit_mask: (batch_size, num_visits) - True per visite valide
        """
        batch_size, num_visits, hidden_size = visit_representations.shape
        
        # 1. Elabora ogni visita individualmente
        visit_encoded = self.visit_encoder(visit_representations)
        # Shape: (batch_size, num_visits, hidden_size//2)
        
        # 2. Applica LSTM per catturare dipendenze temporali
        lstm_output, (hidden, cell) = self.temporal_lstm(visit_encoded)
        # Shape: (batch_size, num_visits, hidden_size//2)
        
        # 3. Applica attention per pesare le visite
        if visit_mask is not None:
            # Converti mask in formato per attention (True = non mascherare)
            attention_mask = ~visit_mask  # Inverti perché attention vuole False per valori validi
        else:
            attention_mask = None
            
        attended_output, attention_weights = self.attention(
            query=lstm_output,
            key=lstm_output,
            value=lstm_output,
            key_padding_mask=attention_mask
        )
        
        # 4. Aggrega le informazioni (media pesata escludendo padding)
        if visit_mask is not None:
            # Maschera i padding prima di fare la media
            mask_expanded = visit_mask.unsqueeze(-1).float()
            attended_masked = attended_output * mask_expanded
            aggregated = attended_masked.sum(dim=1) / mask_expanded.sum(dim=1)
        else:
            aggregated = attended_output.mean(dim=1)
        
        # 5. Classificazione finale
        logits = self.classifier(aggregated)
        
        return logits, attention_weights


class ClinicalLLMClassifier(nn.Module):
    """
    Modello completo che combina un LLM base con classificazione temporale
    """
    def __init__(self, model_name: str, max_visits: int = 10):
        super().__init__()
        
        # Carica il modello base
        self.config = AutoConfig.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModel.from_pretrained(model_name)
        
        # Assicurati che il tokenizer abbia un pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Testa di classificazione temporale
        self.classification_head = TemporalClassificationHead(
            hidden_size=self.config.hidden_size,
            num_visits=max_visits,
            num_classes=2
        )
        
        # Token speciali per delimitare le visite
        self.visit_start_token = "<VISIT_START>"
        self.visit_end_token = "<VISIT_END>"
        
        # Aggiungi token speciali al vocabulary
        special_tokens = [self.visit_start_token, self.visit_end_token]
        self.tokenizer.add_tokens(special_tokens)
        self.llm.resize_token_embeddings(len(self.tokenizer))
    
    def encode_visits(self, visit_texts: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Codifica le visite per ogni paziente
        
        Args:
            visit_texts: Lista di liste, dove ogni lista interna contiene i testi delle visite per un paziente
        
        Returns:
            visit_representations: (batch_size, max_visits, hidden_size)
            visit_mask: (batch_size, max_visits)
        """
        batch_size = len(visit_texts)
        max_visits = max(len(visits) for visits in visit_texts)
        hidden_size = self.config.hidden_size
        
        # Inizializza tensori
        visit_representations = torch.zeros(batch_size, max_visits, hidden_size)
        visit_mask = torch.zeros(batch_size, max_visits, dtype=torch.bool)
        
        for patient_idx, patient_visits in enumerate(visit_texts):
            for visit_idx, visit_text in enumerate(patient_visits):
                # Aggiungi token speciali
                formatted_text = f"{self.visit_start_token} {visit_text} {self.visit_end_token}"
                
                # Tokenizza
                inputs = self.tokenizer(
                    formatted_text,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=512
                )
                
                # Ottieni rappresentazione dal LLM
                with torch.no_grad():
                    outputs = self.llm(**inputs)
                    # Usa il hidden state medio come rappresentazione della visita
                    visit_repr = outputs.last_hidden_state.mean(dim=1)  # (1, hidden_size)
                    
                visit_representations[patient_idx, visit_idx] = visit_repr.squeeze(0)
                visit_mask[patient_idx, visit_idx] = True
        
        return visit_representations, visit_mask
    
    def forward(self, visit_texts: List[List[str]]):
        """
        Forward pass completo
        """
        # 1. Codifica le visite
        visit_representations, visit_mask = self.encode_visits(visit_texts)
        
        # 2. Classificazione temporale
        logits, attention_weights = self.classification_head(
            visit_representations, visit_mask
        )
        
        return {
            'logits': logits,
            'attention_weights': attention_weights,
            'visit_mask': visit_mask
        }


# Esempio di utilizzo
def example_usage():
    """
    Esempio pratico di come usare il modello
    """
    
    # Inizializza il modello
    model = ClinicalLLMClassifier("microsoft/DialoGPT-medium", max_visits=5)
    
    # Dati di esempio
    patient_data = [
        [  # Paziente 1
            "Visita 1: Paziente presenta febbre alta, tosse persistente",
            "Visita 2: Sintomi peggiorati, difficoltà respiratorie",
            "Visita 3: Ricovero in terapia intensiva"
        ],
        [  # Paziente 2
            "Visita 1: Controllo di routine, parametri nella norma",
            "Visita 2: Lieve aumento pressione arteriosa",
            "Visita 3: Pressione stabilizzata con terapia",
            "Visita 4: Paziente in buone condizioni generali"
        ]
    ]
    
    # Labels (0 = sopravvive, 1 = muore)
    labels = torch.tensor([1, 0])  # Paziente 1 muore, Paziente 2 sopravvive
    
    # Forward pass
    model.train()
    outputs = model(patient_data)
    
    # Calcola loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(outputs['logits'], labels)
    
    print(f"Logits: {outputs['logits']}")
    print(f"Predictions: {torch.argmax(outputs['logits'], dim=1)}")
    print(f"Loss: {loss.item()}")
    
    # Visualizza attention weights
    attention = outputs['attention_weights'][0]  # Primo paziente
    print(f"Attention weights shape: {attention.shape}")
    
    return model, outputs, loss

if __name__ == "__main__":
    model, outputs, loss = example_usage()