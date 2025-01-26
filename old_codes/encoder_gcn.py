# %%
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool
import torch
from torch_geometric.data import Data, DataLoader
import numpy as np
import pickle
from skimage import measure

from sklearn.model_selection import train_test_split
import torch_trainer
import os
import matplotlib.pyplot as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# %%

with open('./training_data/geo_sdf_randv_1.pkl', "rb") as f:
  geo_data = pickle.load(f)
vertices_all=geo_data['vertices']
inner_loops_all=geo_data['inner_loops']
out_loop_all=geo_data['out_loop']
points_cloud_all=geo_data['points_cloud']
sdf_all=np.array(geo_data['sdf'],dtype=np.float32)
x_grids=geo_data['x_grids'].astype(np.float32)
y_grids=geo_data['y_grids'].astype(np.float32)
# %%

def generate_graph(vertices,inner_loops,out_loop):
    # Create edges between consecutive points, closing the loop
    edges = []
    for i in range(len(out_loop)-1):
      edges.append((out_loop[i], out_loop[i+1]))
    for loop in inner_loops:
      for i in range(len(loop)-1):
        edges.append((loop[i], loop[i+1]))

    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()
    # Create node features (e.g., (x, y) coordinates)
    node_features = torch.tensor(vertices[:,:2], dtype=torch.float)
    return Data(x=node_features, edge_index=edges)

# %%
class implicit_sdf():
    def __init__(self, input_size, hidden_sizes, output_size):
        """the output size is not 1, in order the match total number of parameters
        to the output size of the encoder.
        the average of the output is taken as the final output,
        so the final output size is 1"""

        # Create a list of (weight size, bias size, activation function) tuples
        self.layers = []
        in_size = input_size

        for hidden_size in hidden_sizes:
            weight = (hidden_size, in_size)
            bias = (1, hidden_size)
            self.layers.append((weight, bias, nn.ReLU()))
            in_size = hidden_size
        weight = (output_size, in_size)
        bias = (1, output_size)
        self.layers.append((weight, bias, nn.Identity()))
        # the total number of parameters must match the output size of the encoder
        self.num_para = sum([w[0] * w[1] + b[1] for w, b, _ in self.layers])

    def __call__(self, x, params_all):
        s = 0
        x = x.expand(params_all.shape[0], -1, -1)
        for w_size, b_size, af in self.layers:
            layer_weight_size = w_size[0] * w_size[1]
            weight = params_all[:, s: (
                s + layer_weight_size)].reshape(-1, *w_size)
            s += layer_weight_size
            layer_bias_size = b_size[1]
            bias = params_all[:, s: (s + layer_bias_size)].reshape(-1, *b_size)
            s += layer_bias_size
            # b is the batch size
            # p is the number of points to evaluate sign distance
            # i is the input size of linear layer
            # o is the output size of linear layer
            x = torch.einsum("bpi,boi->bpo", x, weight)
            x = x + bias
            x = af(x)
        x = torch.mean(x, dim=-1)
        return x



# %%
# Define a simple GNN encoder model
class EncoderGNNModel(torch.nn.Module):
    def __init__(self, num_node_features, graph_hid_dims,fcn_hid_dims, output_dim):
        super().__init__()
        self.layers=nn.ModuleList()
        self.layers.append(GCNConv(num_node_features, graph_hid_dims[0]))
        self.layers.append(torch.nn.ReLU())
        for i in range(1,len(graph_hid_dims)):
            self.layers.append(GCNConv(graph_hid_dims[i-1], graph_hid_dims[i]))
            self.layers.append(torch.nn.ReLU())
        self.layers.append(torch.nn.Linear(graph_hid_dims[-1], fcn_hid_dims[0]))
        self.layers.append(torch.nn.ReLU())
        for i in range(1,len(fcn_hid_dims)):
            self.layers.append(torch.nn.Linear(fcn_hid_dims[i-1], fcn_hid_dims[i]))
            self.layers.append(torch.nn.ReLU())
        self.fc = torch.nn.Linear(fcn_hid_dims[-1], output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for layer in self.layers:
          if isinstance(layer, GCNConv):
              x = layer(x, edge_index)
          else:
              x = layer(x)
        x = global_mean_pool(x, data.batch)  # Pooling layer
        x = self.fc(x)
        return x


# %%
# Generate multiple surfaces
graphs=[generate_graph(vertices_all[i],inner_loops_all[i],out_loop_all[i]) for i in range(len(vertices_all))]
# %%
# Attach sign distance and grid points

grid_points = np.vstack([x_grids.ravel(), y_grids.ravel()]).T
SDFs=torch.tensor(sdf_all)
grid_coor=torch.tensor(grid_points).to(device)
for i in range(len(graphs)):
    graphs[i].y=SDFs[i]
#%%
graph_train, graph_test, SDF_train, SDF_test = train_test_split(graphs, SDFs, test_size=0.2, random_state=42)

for i in range(len(graph_train)):
    graph_train[i].y=SDF_train[i]
for i in range(len(graph_test)):
    graph_test[i].y=SDF_test[i]
train_loader = DataLoader(graph_train, batch_size=128, shuffle=True)
test_loader = DataLoader(graph_test, batch_size=2048, shuffle=False)


# train_loader = DataLoader(graphs[:1], batch_size=128, shuffle=True)
# test_loader = DataLoader(graphs[-1000:], batch_size=2048, shuffle=False)
# %%

class TRAINER(torch_trainer.TorchTrainer):
  def __init__(self, model, device,sdf):
    super().__init__(model, device)
    self.sdf=sdf
  def evaluate_losses(self, data):
        # num=0
        # for param in self.model.parameters():
        #     if param.grad is not None:
        #         num+=param.numel()
        # print("number of parameters with grad:",num)
        params = self.model(data)
        sdf_pred = self.sdf(grid_coor, params)
        loss = self.loss_fn(sdf_pred, data.y.view(-1,grid_coor.shape[0]))

        loss_dic = {"loss": loss.item()}
        return loss, loss_dic

# %%

sdf_NN = implicit_sdf(input_size=2, hidden_sizes=[12]*7, output_size=4)
print("Total num parameters of implicit sdf: ", sdf_NN.num_para)
output_dim_encoder=32*32
if output_dim_encoder != sdf_NN.num_para:
    raise ValueError("output_dim_encoder must match the number of parameters of sdf_NN")
geo_encoder = EncoderGNNModel(num_node_features=2, graph_hid_dims=[64]*2,fcn_hid_dims=[100]*4, output_dim=output_dim_encoder)  # Assuming output is a scalar
print("number parameters of geo_encoder:", sum(p.numel() for p in geo_encoder.parameters()))



trainer = TRAINER(geo_encoder, device,sdf_NN)
trainer.compile(optimizer=torch.optim.Adam(
    geo_encoder.parameters(), lr=1e-3), loss=nn.MSELoss())
filebase = "./saved_model/geo_encoder"
os.makedirs(filebase, exist_ok=True)
checkpoint_fname = os.path.join(filebase, "model.ckpt")
checkpoint = torch_trainer.ModelCheckpoint(
    checkpoint_fname, monitor="val_loss", save_best_only=True
)
# %%
# trainer.load_weights(checkpoint_fname, device)
# h = trainer.load_logs(filebase)
# h = trainer.fit(train_loader, val_loader=test_loader,
#                 epochs=2000, callbacks=checkpoint, print_freq=1)
# trainer.save_logs(filebase)

# %%
trainer.load_weights(checkpoint_fname, device)
h = trainer.load_logs(filebase)
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.plot(h["loss"], label="loss")
ax.plot(h["val_loss"], label="val_loss")
ax.legend()
ax.set_yscale("log")


#%%
test_data=next(iter(test_loader))
test_data=test_data.to(device)
para_test=geo_encoder(test_data)
sd_pred=trainer.sdf(grid_coor,para_test).cpu().detach().numpy()
sd_true=test_data.y.view(-1,grid_coor.shape[0]).cpu().numpy()
error_s=np.linalg.norm(sd_pred-sd_true,axis=1)/np.linalg.norm(sd_true,axis=1)

fig = plt.figure(figsize=(4.8, 3.6))
ax = plt.subplot(1, 1, 1)
_=ax.hist(error_s, bins=20)

# %%

# # %%
sort_idx = np.argsort(error_s)
min_index = sort_idx[0]
max_index = sort_idx[-1]
median_index = sort_idx[len(sort_idx) // 2]
# # Print the indexes
print("Index for minimum geo:", min_index,
      "with error", error_s[min_index])
print("Index for maximum geo:", max_index,
      "with error", error_s[max_index])
print("Index for median geo:", median_index,
      "with error", error_s[median_index])
min_median_max_index = np.array([min_index, median_index, max_index])
nr, nc = 1, 3
fig = plt.figure(figsize=(nc*4.8, nr*3.6))
for i, index in enumerate(min_median_max_index):

    ax = plt.subplot(nr, nc, i+1)
    sd_pred_i = sd_pred[index].reshape(x_grids.shape)
    sd_true_i = sd_true[index].reshape(x_grids.shape)
    pred_geo = measure.find_contours(sd_pred_i, 0, positive_orientation='high')
    true_geo = measure.find_contours(sd_true_i, 0, positive_orientation='high')
    for c, contour in enumerate(true_geo):
        if c == 0:
            ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2, label="Truth")
        else:
            ax.plot(contour[:, 1], contour[:, 0], 'r', linewidth=2)
    for c, contour in enumerate(pred_geo):
        if c == 0:
            ax.plot(contour[:, 1], contour[:, 0], 'b', linewidth=2, label="Predicted")
        else:
            ax.plot(contour[:, 1], contour[:, 0], 'b', linewidth=2)

    ax.legend()
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")

    plt.tight_layout()

# %%
