from router_model.regression_models import TruncatedModel, load_tokenizer
from router_model.config import RouterModelConfig

if __name__ == "__main__":
    training_config = RouterModelConfig(
        dropout_rate=0.1,
        classifier_dropout=True,
        classifier_type="linear"
    )


    model = TruncatedModel.load_model_from_checkpoint("router_model/model_checkpoints/model_deberta_20260101-163459.pth", 
    model_name="deberta",
    pooling_strategy="cls",
    training_config=training_config)