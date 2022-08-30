from schemas.main import Args
from utils import (configure_logger,
                   get_args,
                   initialize_datamodule,
                   initialize_wandb_logger,
                   initialize_trainer,
                   CNN,
                   log_args,
                   initialize_model)


def main():
    args = get_args(args_class=Args)
    configure_logger(args.env.path, level='DEBUG')
    log_args(args)

    datamodule = initialize_datamodule(args.data, args.opt.batch_size)
    wandb_logger = initialize_wandb_logger(args)
    model = initialize_model(args, wandb_logger, model_class=CNN)
    trainer = initialize_trainer(args.env, args.opt, wandb_logger)
    trainer.fit(model, datamodule=datamodule)


if __name__ == '__main__':
    main()
