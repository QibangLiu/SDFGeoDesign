# %%
from torch.utils.data import DataLoader
import argparse
import os
from skimage import measure
import torch
import numpy as np
import matplotlib.pyplot as plt
current_work_path = os.getcwd()
current_file_dir = os.path.dirname(os.path.abspath(__file__))
if __package__:
    from .geoencoder import LoadGeoEncoderModel
    from .configs import models_configs, LoadData
    from .modules.UNets import UNet, UNetTimeStep
    from .modules.diffusion import GaussianDiffusion
    from .trainer import torch_trainer
else:
    from geoencoder import LoadGeoEncoderModel
    from configs import models_configs, LoadData
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
                                    num_res_blocks=1, total_timesteps=500):

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
    )
    trainable_params = sum(p.numel()
                           for p in inv_Unet.parameters() if p.requires_grad)
    print(
        f"Total number of trainable parameters of inverse Unet of diffusion: {trainable_params}")

    gaussian_diffusion = GaussianDiffusion(img_shape=img_shape,
                                           timesteps=total_timesteps)

    return inv_Unet, gaussian_diffusion


def TrainDiffusionInverseModel(inv_Unet, gaussian_diffusion, geo_encoder,
                               filebase, train_flag, train_loader,
                               test_loader, epochs=300, lr=1e-3):
    total_timesteps = gaussian_diffusion.timesteps

    class TorchTrainer(torch_trainer.TorchTrainer):
        def __init__(self, model, device, filebase, geo_encoder):
            super().__init__(model, device, filebase)
            self.geo_encoder = geo_encoder
            for param in geo_encoder.parameters():
                param.requires_grad = False
            self.geo_encoder.eval()

        def evaluate_losses(self, data):
            '''custom loss'''
            pc = data[0].to(self.device)
            labels = data[1].to(self.device)
            latent = self.geo_encoder(pc)  # (Nb,lat_dim,out_c)
            latent = latent.unsqueeze(1)  # (Nb,1,lat_dim,out_c)
            batch_size = latent.shape[0]
            # random generate mask
            z_uncound = torch.rand(batch_size)
            batch_mask = (z_uncound > 0.2).int().to(device)

            # sample t uniformally for every example in the batch
            t = torch.randint(0, total_timesteps, (batch_size,),
                              device=device).long()

            loss = gaussian_diffusion.train_losses(
                self.models[0], latent, t, labels, batch_mask
            )
            loss_tracker = {"loss": loss.item()}
            return loss, loss_tracker

    trainer = TorchTrainer(inv_Unet, device, filebase, geo_encoder)

    checkpoint = torch_trainer.ModelCheckpoint(
        monitor="loss", save_best_only=True)

    optimizer = torch.optim.Adam(trainer.parameters(), lr=lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.7, patience=40)
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
            train_loader, val_loader=None, epochs=epochs, print_freq=1
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


def EvaluateDiffusionInverseModel(fwd_model, inv_Unet, gaussian_diffusion, sdf_NN, grid_coor,
                                  Ytarget, sdf_inv_scaler, stress_inv_scaler,
                                  num_sol=10, plot_flag=False):
    Ytarget = Ytarget.to(device)
    labels = Ytarget.repeat(num_sol, 1)
    latent = gaussian_diffusion.sample(
        inv_Unet, labels, w=2, clip_denoised=False
    )
    latent = torch.tensor(latent).to(device)
    with torch.no_grad():
        Ypred = fwd_model(latent)
    Ypred_inv = stress_inv_scaler(Ypred.cpu().detach().numpy())
    Ytarg_inv = stress_inv_scaler(labels.cpu().detach().numpy())
    sdf_grid = sdf_NN(grid_coor, latent.squeeze())
    Xpred_inv = sdf_inv_scaler(sdf_grid.cpu().detach().numpy())
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

    if plot_flag:
        strain = np.linspace(0, 0.2, 51)
        legends = ["best", "33\%", "66\%", "worst"]
        fig = plt.figure(figsize=(4.8 * 5, 3.6))
        ax = plt.subplot(1, 5, 1)
        ax.plot(strain, Ytarg_inv[0], label="target")
        for i, v in enumerate(evl_ids):
            ax.plot(strain, Ypred_inv[v], label=legends[i])
        ax.legend()
        ax.set_xlabel(r"$\varepsilon~[\%]$")
        ax.set_ylabel(r"$\sigma~[MPa]$")
        for i, v in enumerate(evl_ids):
            contours = measure.find_contours(
                Xpred_inv[v], 0, positive_orientation="high")
            ax = plt.subplot(1, 5, i + 2)
            l_style = ["r-", "b--"]
            holes = []
            for j, contour in enumerate(contours):
                contour = (contour - 10) / 100
                x, y = contour[:, 1], contour[:, 0]
                if j == 0:
                    ax.fill(
                        x,
                        y,
                        alpha=1.0,
                        edgecolor="black",
                        facecolor="cyan",
                        label="Outer Boundary",
                    )
                else:
                    ax.fill(x, y, alpha=1.0, edgecolor="black",
                            facecolor="white", label="Hole")
                # ax.grid(True)
                ax.axis("off")
                ax.axis("equal")  # Keep aspect ratio square


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_flag", type=str, default="start")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    args, unknown = parser.parse_known_args()
    print(vars(args))

    configs = models_configs(diffusion_from_lattent=True)
    filebase = configs["InvDiffusion"]["filebase"]
    model_args = configs["InvDiffusion"]["model_args"]
    geo_encoder_filebase = configs["GeoEncoder"]["filebase"]
    geo_encoder_model_args = configs["GeoEncoder"]["model_args"]
    fwd_filebase = configs["ForwardModel"]["filebase"]
    fwd_model_args = configs["ForwardModel"]["model_args"]
    print(f"\n\nInvDiffusion Filebase: {filebase}, model_args:")
    print(model_args)
    print(f"\n\nGeoEncoder Filebase: {geo_encoder_filebase}, model_args:")
    print(geo_encoder_model_args)
    print(f"\n\nForwardModel Filebase: {fwd_filebase}, model_args:")
    print(fwd_model_args)

    train_dataset, test_dataset, grid_coor, sdf_inv_scaler, stress_inv_scaler = LoadData(
        seed=42)
    grid_coor = grid_coor.to(device)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    geo_encoder, sdf_NN = LoadGeoEncoderModel(
        geo_encoder_filebase, geo_encoder_model_args)

    inv_Unet, gaussian_diffusion = DiffusionInverseModelDefinition(
        **model_args)
    trainer = TrainDiffusionInverseModel(inv_Unet, gaussian_diffusion, geo_encoder, filebase, args.train_flag,
                                         train_loader, test_loader, epochs=args.epochs, lr=args.learning_rate)


# %%
