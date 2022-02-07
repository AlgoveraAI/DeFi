from fastai.tabular.all import *
import seaborn as sns
from tqdm import tqdm
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import RMSE, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters

def get_train_data(df1, target_token, feature_cols, max_encoder_length=24, min_pred=6, max_pred=12):

    training_cutoff = int(df1['time_idx'].max()*0.8)

    training = TimeSeriesDataSet(
        df1[lambda x: x.time_idx <= training_cutoff],
        group_ids=['Dummy'],
        time_idx="time_idx",
        target=f"{target_token}_Target",
        min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
        max_encoder_length=max_encoder_length,
        min_prediction_length=min_pred,
        max_prediction_length=max_pred,
        time_varying_known_categoricals=[],
        time_varying_unknown_categoricals=[],
        time_varying_known_reals=feature_cols,
    #    time_varying_unknown_reals=[f"{target_token}_Target"],
        time_varying_unknown_reals=[],
        allow_missing_timesteps=True,
        target_normalizer=None
    )

    validation = TimeSeriesDataSet(
    df1[lambda x: x.time_idx > training_cutoff],
    group_ids=['Dummy'],
    time_idx="time_idx",
    target=f"{target_token}_Target",
    min_encoder_length=max_encoder_length // 2,  # keep encoder length long (as it is in the validation set)
    max_encoder_length=max_encoder_length,
    min_prediction_length=min_pred,
    max_prediction_length=max_pred,
    time_varying_known_categoricals=[],
    time_varying_unknown_categoricals=[],
    time_varying_known_reals=feature_cols,
#    time_varying_unknown_reals=[f"{target_token}_Target"],
    time_varying_unknown_reals=[],
    allow_missing_timesteps=True,
    target_normalizer=None
)

    # create dataloaders for model
    batch_size = 32  # set this between 32 to 128
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=0)

    return training, train_dataloader, val_dataloader

def train_tft(training, train_dataloader, val_dataloader, max_epochs=30, dropout=0.1, attention_head_size=1):
    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        gpus=1,
        weights_summary="top",
        gradient_clip_val=0.1,
        limit_train_batches=30,  
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )

    tft = TemporalFusionTransformer.from_dataset(
        training,
        learning_rate=0.03,
        hidden_size=16,
        attention_head_size=attention_head_size,
        dropout=dropout,
        hidden_continuous_size=6,
        output_size=7,
        loss=QuantileLoss(),
        log_interval=10,  # uncomment for learning rate finder and otherwise, e.g. to 10 for logging every 10 batches
        reduce_on_plateau_patience=4,
    )

    trainer.fit(
        tft,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader,
    )

    return trainer