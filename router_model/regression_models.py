import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from transformers import DistilBertModel, AutoModel, AutoTokenizer, DistilBertTokenizer
from .config import RouterModelConfig


def load_tokenizer(model_name: str, context_window: int = 512):
    """
    Load tokenizer for the specified model.
    
    Args:
        model_name: Name of the model (e.g., "deberta", "distilbert", "bert", "tinybert")
        context_window: Maximum context window length
    
    Returns:
        tokenizer: Loaded tokenizer
    """
    if model_name == "distilbert":
        tokenizer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased",
            max_length=context_window,
            truncation_side="left",
            clean_up_tokenization_spaces=False
        )
    elif model_name == "deberta":
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/deberta-v3-large",
            max_length=context_window,
            truncation_side="left",
            clean_up_tokenization_spaces=False
        )
    elif model_name == "tinybert":
        tokenizer = AutoTokenizer.from_pretrained(
            "huawei-noah/TinyBERT_General_6L_768D",
            max_length=context_window,
            truncation_side="left",
            clean_up_tokenization_spaces=False
        )
    elif model_name == "bert":
        tokenizer = AutoTokenizer.from_pretrained(
            'microsoft/deberta-v3-base',
            max_length=context_window,
            truncation_side="left",
            clean_up_tokenization_spaces=False
        )
    else:
        raise ValueError(f"Invalid model name: {model_name}")
    
    return tokenizer


class TruncatedModel(nn.Module):
    def __init__(self, num_outputs, num_classes, model_name, pooling_strategy, router_model_config):
        self.pooling_strategy = pooling_strategy
        self.router_model_config = router_model_config

        super().__init__()
        # Load model with bfloat16 precision to save memory (H100 supports BF16 natively)
        if model_name == "deberta":
            self.transformer = AutoModel.from_pretrained("microsoft/deberta-v3-large", dtype=torch.bfloat16)
        elif model_name == "distilbert":
            self.transformer = DistilBertModel.from_pretrained("distilbert-base-uncased", dtype=torch.bfloat16)
        elif model_name == "tinybert":
            self.transformer = AutoModel.from_pretrained("huawei-noah/TinyBERT_General_6L_768D", dtype=torch.bfloat16)
        elif model_name == "bert":
            self.transformer = AutoModel.from_pretrained("microsoft/deberta-v3-base", dtype=torch.bfloat16)
        else:
            raise ValueError(f"Invalid model name: {model_name}")

        # Freeze the layers
        if self.router_model_config.freeze_layers and model_name != "distilbert":    
            for i, layer in enumerate(self.transformer.encoder.layer):
                if i < self.router_model_config.layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    # Explicitly ensure unfrozen layers have requires_grad=True
                    for param in layer.parameters():
                        param.requires_grad = True

        elif self.router_model_config.freeze_layers and model_name == "distilbert":
            for i, layer in enumerate(self.transformer.transformer.layer):
                if i < self.router_model_config.layers_to_freeze:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    # Explicitly ensure unfrozen layers have requires_grad=True
                    for param in layer.parameters():
                        param.requires_grad = True

        
        # Freeze the embedding layer 
        if self.router_model_config.freeze_embedding:
            if model_name == "deberta":
                embedding_module = self.transformer.embeddings
            elif model_name == "distilbert":
                embedding_module = self.transformer.transformer.embeddings
            elif model_name == "tinybert":
                embedding_module = self.transformer.embeddings
            elif model_name == "bert":
                embedding_module = self.transformer.embeddings
            for param in embedding_module.parameters():
                param.requires_grad = False

        if self.pooling_strategy == "attention":
            self.attention_vector= nn.Parameter(torch.randn(self.transformer.config.hidden_size))
        
        # Create classifier based on configuration
        if self.router_model_config.classifier_type == "linear":
            self.classifier = nn.Linear(self.transformer.config.hidden_size, num_outputs)
        elif self.router_model_config.classifier_type == "mlp":
            self.classifier = nn.Sequential(
                nn.Linear(self.transformer.config.hidden_size, self.router_model_config.mlp_hidden_size),
                nn.ReLU(),
                nn.Dropout(self.router_model_config.dropout_rate),
                nn.Linear(self.router_model_config.mlp_hidden_size, num_outputs)
            )
        else:
            raise ValueError(f"Invalid classifier_type: {self.router_model_config.classifier_type}. Must be 'linear' or 'mlp'")
            
        if self.router_model_config.classifier_dropout and self.router_model_config.classifier_type == "linear":
            self.dropout = nn.Dropout(self.router_model_config.dropout_rate)
        else:
            self.dropout = None
        
        # Ensure classifier is in float32 for numerical stability
        self.classifier = self.classifier.float()
        
        # Explicitly ensure classifier parameters are trainable
        for param in self.classifier.parameters():
            param.requires_grad = True

    
    def forward(self, input_ids, attention_mask):
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state     # (batch_size, seq_len, hidden)
        if self.pooling_strategy == "cls":
            cls_embedding = last_hidden_state[:, 0]       # Use [CLS] token representation
        elif self.pooling_strategy == "last":
            cls_embedding = last_hidden_state[:, -1]      # Use last token representation
        elif self.pooling_strategy == "mean":
            masked_hidden_state  = last_hidden_state * torch.unsqueeze(attention_mask, -1)
            cls_embedding = masked_hidden_state.sum(dim=1) / attention_mask.sum(dim=1, keepdim=True) 
        elif self.pooling_strategy == "max":
            cls_embedding = last_hidden_state.max(dim=1).values  
        elif self.pooling_strategy == "attention":
            attention_weights = torch.matmul(last_hidden_state, self.attention_vector)
            attention_weights = F.softmax(attention_weights, dim=1)
            cls_embedding = torch.sum(attention_weights.unsqueeze(2) * last_hidden_state, dim=1)
        else:
            raise ValueError(f"Invalid pooling strategy: {self.pooling_strategy}")
        if self.dropout is not None:    
            cls_embedding = self.dropout(cls_embedding)
        # Convert to float32 for classifier (more stable for final layer)
        cls_embedding = cls_embedding.float()
        raw_output = self.classifier(cls_embedding)
        
        return raw_output
    
    @classmethod
    def load_model_from_checkpoint(
        cls,
        model_path: str,
        model_name: str,
        pooling_strategy: str,
        num_outputs: int = 1,
        num_classes: int = 2,
        router_model_config: RouterModelConfig = None,
        device: str = 'cuda'
    ):
        """
        Load a TruncatedModel from a checkpoint file.
        
        Args:
            model_path: Path to the model checkpoint (.pth file)
            model_name: Name of the model (e.g., "deberta", "distilbert", "bert", "tinybert")
            pooling_strategy: Pooling strategy used for router model (e.g., "cls", "mean", "max", "attention")
            num_outputs: Number of output dimensions (default: 1 for binary classification)
            num_classes: Number of classes (default: 2)
            router_model_config: RouterModelConfig object (if None, creates default config)
            device: Device to load model on
        
        Returns:
            model: Loaded TruncatedModel instance
        """
        # Load model state dict
        state_dict = torch.load(model_path, map_location=device)
        
        # Create default training config if not provided
        if router_model_config is None:
            router_model_config = RouterModelConfig(
                dropout_rate=0.1,
                classifier_dropout=True,
                classifier_type="linear"
            )
        
        # Initialize model with same config as training
        model = cls(
            num_outputs=num_outputs,
            num_classes=num_classes,
            model_name=model_name,
            pooling_strategy=pooling_strategy,
            router_model_config=router_model_config
        )
        
        # Load state dict
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        
        return model

class TextRegressionDataset(Dataset):
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
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Remove batch dimension
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        # Check if label is a tensor
        if isinstance(label, torch.Tensor):
            item['labels'] = label
        else:
            item['labels'] = torch.tensor(label, dtype=torch.float)
        return item