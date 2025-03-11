# %%
from torch.utils.data import DataLoader
import argparse
import os
from skimage import measure
import torch
import numpy as np
import matplotlib.pyplot as plt
if __package__:
    from .NOT_SS import LoadNOTModel
    from . import configs
    from .modules.UNets import UNet, UNetTimeStep
    from .modules.diffusion import GaussianDiffusion
    from .trainer import torch_trainer
else:
    from NOT_SS import LoadNOTModel
    import configs
    from modules.UNets import UNet, UNetTimeStep
    from modules.diffusion import GaussianDiffusion
    from trainer import torch_trainer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%


def DiffusionInverseModelDefinition(img_shape=(1, 128, 128),
                                    channel_multpliers=[1, 2, 4, 8],
                                    has_attention=[False, False, True, True],
                                    fist_conv_channels=32, num_heads=4,
                                    norm_groups=16,
                                    num_res_blocks=1, dropout=None, total_timesteps=500):

    label_dim = 51
    inv_Unet = UNetTimeStep(
        img_shape=img_shape,
        label_dim=label_dim,
        one_hot=False,
        first_conv_channels=fist_conv_channels,
        channel_mutipliers=channel_multpliers,
        has_attention=has_attention,
        num_heads=num_heads,
        num_res_blocks=num_res_blocks,
        norm_groups=norm_groups,
        dropout=dropout,
    )
    trainable_params = sum(p.numel()
                           for p in inv_Unet.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters of inverse Unet of diffusion: {trainable_params}")

    gaussian_diffusion = GaussianDiffusion(img_shape=img_shape,
                                           timesteps=total_timesteps)

    return inv_Unet, gaussian_diffusion


def TrainDiffusionInverseModel(inv_Unet, gaussian_diffusion,
                               filebase, train_flag, train_loader,
                               test_loader, epochs=300, lr=1e-3):
    total_timesteps = gaussian_diffusion.timesteps

    class TorchTrainer(torch_trainer.TorchTrainer):
        def __init__(self, model, device, filebase):
            super().__init__(model, device, filebase)

        def evaluate_losses(self, data):
            '''custom loss'''
            labels = data[1].to(self.device)  # (B, 51)
            normalized_sdf = data[0].to(self.device)  # (B, 1,120,120)
            batch_size = normalized_sdf.shape[0]
            # random generate mask
            z_uncound = torch.rand(batch_size)
            batch_mask = (z_uncound > 0.1).int().to(device)
            # sample t uniformally for every example in the batch
            t = torch.randint(0, total_timesteps, (batch_size,),
                              device=device).long()
            loss = gaussian_diffusion.train_losses(
                self.models[0], normalized_sdf, t, labels, batch_mask
            )
            loss_tracker = {"loss": loss.item()}
            return loss, loss_tracker

    trainer = TorchTrainer(inv_Unet, device, filebase)

    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="loss", save_best_only=True)

    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=20)
    trainer.compile(
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        checkpoint=checkpoint,
        scheduler_metric_name="loss",
    )
    if not train_flag == "start":
        trainer.load_weights(device=device)
        h = trainer.load_logs()

    if train_flag == "continue" or train_flag == "start":
        h = trainer.fit(
            train_loader, val_loader=test_loader, epochs=epochs, print_freq=1
        )
    trainer.save_logs()
    trainer.load_weights(device=device)

    return trainer


def LoadDiffusionInverseModel(file_base, model_args):
    model_path = os.path.join(file_base, "model.ckpt")
    inv_Unet, gaussian_diffusion = DiffusionInverseModelDefinition(
        **model_args)
    state_dict = torch.load(model_path, map_location=device, weights_only=True)
    inv_Unet.load_state_dict(state_dict)
    inv_Unet.to(device)
    inv_Unet.eval()
    return inv_Unet, gaussian_diffusion


def EvaluateDiffusionInverseModel(fwd_model, inv_Unet, gaussian_diffusion,
                                  Ytarget, sdf_inv_scaler, stress_inv_scaler,
                                  num_sol=10):
    Ytarget = Ytarget.to(device)
    labels = Ytarget.repeat(num_sol, 1)
    sdf = gaussian_diffusion.sample(
        inv_Unet, labels, w=6, clip_denoised=False
    )
    sdf = torch.tensor(sdf).to(device)
    strain = torch.tensor(np.linspace(0, 1, 51), dtype=torch.float32)
    strain = strain[:, None].to(device)
    with torch.no_grad():
        Ypred = fwd_model(strain, sdf)
    Ypred_inv = stress_inv_scaler(Ypred.cpu().detach().numpy())
    Ytarg_inv = stress_inv_scaler(labels.cpu().detach().numpy())
    Xpred_inv = sdf_inv_scaler(sdf.cpu().detach().numpy())
    L2error = np.linalg.norm(Ypred_inv - Ytarg_inv, axis=1) / \
        np.linalg.norm(Ytarg_inv, axis=1)
    sorted_idx = np.argsort(L2error)
    mean, std = np.mean(L2error), np.std(L2error)
    print(f"Mean L2 error of the diffusion design results: {mean}, std: {std}")
    evl_ids = np.array([
        sorted_idx[0],
        sorted_idx[int(len(sorted_idx) * 0.33)],
        sorted_idx[int(len(sorted_idx) * 0.66)],
        sorted_idx[-1],
    ], dtype=int)
    for i, idx in enumerate(evl_ids):
        print(f"ID: {idx}, L2 error: {L2error[idx]}")

# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    print(vars(args))

    inv_config = configs.INV_configs()
    filebase = inv_config["filebase"]
    model_args = inv_config["model_args"]
    notss_config = configs.NOTSS_configs()
    filebase_infss = notss_config["filebase"]
    model_args_infss = notss_config["model_args"]
    print(f"\n\nInvDiffusion Filebase: {filebase}, model_args:")
    print(model_args)
    print(f"\n\n NOT ForwardModel Filebase: {filebase_infss}, model_args:")
    print(model_args_infss)

    train_dataset, test_dataset, sdf_inv_scaler, stress_inv_scaler = configs.LoadDataInv()  #
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    not_ss = LoadNOTModel(
        filebase_infss, model_args_infss)

    inv_Unet, gaussian_diffusion = DiffusionInverseModelDefinition(
        **model_args)
    trainer = TrainDiffusionInverseModel(inv_Unet, gaussian_diffusion, filebase, args.train_flag,
                                         train_loader, test_loader, epochs=args.epochs, lr=args.learning_rate)

    id = 2897
    Ytarget = test_dataset[id][1].unsqueeze(0)
    EvaluateDiffusionInverseModel(not_ss, inv_Unet, gaussian_diffusion,
                                  Ytarget, sdf_inv_scaler, stress_inv_scaler,
                                  num_sol=100)

# %%
