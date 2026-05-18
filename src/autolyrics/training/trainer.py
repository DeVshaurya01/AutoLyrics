"""Seq2SeqTrainer wiring with PEFT compatibility."""
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments, EarlyStoppingCallback
from datasets import DatasetDict

from autolyrics.data.collator import DataCollatorSpeechSeq2SeqWithPadding
from autolyrics.evaluation.metrics import make_compute_metrics


def build_trainer(
    model,
    processor,
    ds: DatasetDict,
    training_cfg: dict,
    lora_cfg: dict | None = None,
) -> Seq2SeqTrainer:
    """Wire up Seq2SeqTrainer with correct PEFT compatibility flags."""
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
        apply_specaug=True,
    )

    # Strip metadata columns the model doesn't accept. Required because we
    # set remove_unused_columns=False (for PEFT compatibility), which
    # otherwise leaves text columns like "transcription" in the batch.
    keep = {"input_features", "labels"}
    train_ds = ds["train"].remove_columns(
        [c for c in ds["train"].column_names if c not in keep]
    )
    val_ds = ds["val"].remove_columns(
        [c for c in ds["val"].column_names if c not in keep]
    )

    training_args = Seq2SeqTrainingArguments(**training_cfg)

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        compute_metrics=make_compute_metrics(processor),
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    return trainer
