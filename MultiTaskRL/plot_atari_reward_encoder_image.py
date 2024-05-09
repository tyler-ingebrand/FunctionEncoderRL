import matplotlib.cm as cmx
import cv2
import numpy
import numpy as np
import torch
from matplotlib.colors import LightSource
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from atari_modules.models import TaskAwareCritic, RewardEncoder, RewardEncoderTranslator, BackwardMap, TransformerCritic
from matplotlib import cbook, cm, colors, patches, ticker

from atari_modules.wrappers import make_goalPacman


raise Exception("Change the below lines of code to match a date-time string in your data directory. Then run this script.")
save_dir_date_time_fb = '2023-11-29 06:38:59'
save_dir_date_time_re = '2023-11-17 13:54:13'
save_dir_date_time_rea = '2023-11-29 04:22:37'
save_dir_date_time_transformer = '2023-12-09 22:21:32'


def _preproc_o(obs):
    if type(obs) is not torch.Tensor:
        obs = np.array(obs)
        obs = torch.tensor(obs, dtype=torch.float32, device="cpu")

    if len(obs.shape) == 3:
        obs = obs[None]  # add batch dim of 1 if needed
    obs = obs / 255.
    # obs = torch.transpose(obs, [0, 3, 1, 2])
    obs = obs.permute(0, 3, 1, 2)
    return obs

x_points = [11, 19, 27, 35, 44, 52, 60, 68, 78, 88, 97, 105, 113, 120, 127, 135, 143, 152,]
y_points = [9, 21, 34, 46, 57, 69, 81, 93, 105, 117, 129, 141, 153, 165,]
def nearest_acceptable_grid_point(goal):
    # find nearest acceptable x and y
    x = goal[0]
    y = goal[1]
    x = min(x_points, key=lambda x_point:abs(x_point-x))
    y = min(y_points, key=lambda y_point:abs(y_point-y))
    return [x, y]
def get_reward_function(g, num_actions):
    g = torch.tensor(g, device="cpu") / 170.

    def reward_function(achieved_goal):
        distances = torch.sum((achieved_goal - g) ** 2, dim=1) ** 0.5
        # rewards = torch.where(distances < 0.07, torch.tensor([1.0], device=obs.device), torch.tensor([0.0], device=obs.device))
        rewards = -distances
        rewards = rewards.unsqueeze(1).expand(rewards.shape[0], num_actions).to(torch.float32)
        return rewards

    return reward_function

env_params = {'obs': (1,1),
          'goal': 2,
          'action': 5,
          'max_timesteps': 50,
          }
embed_dim=100



all_goals =  [[11, 9], [19, 9], [27, 9], [35, 9], [50, 9], [58, 9], [66, 9], [76, 9], [86, 9], [95, 9], [104, 9], [112, 9], [127, 9], [135, 9], [143, 9], [152, 9],
          [35, 21], [50, 21], [112, 21], [128, 21], [152, 21],
          [11, 34], [19, 34], [35, 34], [43, 34], [51, 34], [59, 34], [68, 34], [76, 34], [87, 34], [95, 34], [103, 34], [111, 34], [119, 34], [127, 34], [135, 34], [143, 34], [152, 34],
          [20, 46], [35, 46], [60, 46], [103, 46], [129, 46], [144, 46],
          [11, 57], [19, 57], [35, 57], [43, 57], [51, 57], [59, 57], [67, 57], [75, 57], [87, 57], [95, 57], [103, 57], [111, 57], [118, 57], [127, 57], [143, 57], [151, 57],
          [20, 69], [59, 69], [104, 69], [144, 69],
          [20, 82], [27, 82], [35, 82], [43, 82], [43, 82], [51, 82], [59, 82], [103, 82], [111, 82], [128, 82], [135, 82], [142, 82],
          [20, 93], [58, 93], [103, 93], [143, 93],
          [12, 105], [19, 105], [35, 105], [43, 105], [51, 105], [110, 105], [118, 105], [126, 105], [143, 105], [151, 105],
          [19, 117], [35, 117], [50, 117], [50, 117], [67, 117], [94, 117], [110, 117], [127, 117], [143, 117],
          [12, 129], [19, 129], [27, 129], [35, 129], [35, 129], [51, 129], [67, 129], [75, 129], [86, 129], [95, 129], [111, 129], [126, 129], [135, 129], [143, 129], [151, 129],
          [12, 141], [35, 141], [50, 141], [67, 141], [95, 141], [102, 141], [112, 141], [127, 141], [151, 141],
          [12, 153], [35, 153], [67, 153], [95, 153], [127, 153], [151, 153],
          [12, 165], [19, 165], [27, 165], [35, 165], [43, 165], [50, 165], [59, 165], [68, 165], [76, 165], [87, 165], [87, 165], [96, 165], [103, 165],
          [112, 165], [119, 165], [127, 165], [136, 165], [142, 165], [151, 165]]
print(len(all_goals))

x_min, y_min, x_max,y_max = 10000, 10000, -10000, -10000
for goal in all_goals:
    x_min = min(goal[0], x_min)
    y_min = min(goal[1], y_min)
    x_max = max(goal[0], x_max)
    y_max = max(goal[1], y_max)

# create grid over xs and ys
xs = torch.linspace(x_min, x_max, 100)# .to("cuda:0")
ys = torch.linspace(y_min, y_max, 100)# .to("cuda:0")
xs, ys = torch.meshgrid(xs, ys)
xs_flat = xs.reshape(-1)
ys_flat = ys.reshape(-1)
inputs = torch.stack([xs_flat, ys_flat], dim=1)# .to(torch.float32)# .to("cuda:0")
inputs = inputs / 170.


# get function encoder graph
# create the networks
reward_encoder = RewardEncoder(env_params, embed_dim) #.to("cuda:0")
reward_encoder.load_state_dict(torch.load(f'data/{save_dir_date_time_re}/reward_encoder.pt', map_location="cpu"))
all_encoings = []
# Get encodings
for goal in all_goals:
    encoding = reward_encoder(torch.tensor(goal, dtype=torch.float32).unsqueeze(0) / 170.)

    all_encoings.append(encoding.cpu().detach())


# fetch an image from the pacman env. Convert to cv image so we can edit it
env = make_goalPacman()
env.reset()
obs, _, _, _ = env.step(0)
img = env.render(mode='rgb_array')
img = img[:172, :, :]

# want to show cos sim relative to goal 0
goal_to_sim = 0

# compute minimum of all cos similiarities
min_sim = 1
for i in range(len(all_encoings)):
    encoding = all_encoings[i]
    sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)
    min_sim = min(min_sim, sim.item())

# Create a color map. high is 1, low is 0.8
cmap_type = 'YlGn'
jet = plt.get_cmap(cmap_type)
cNorm = colors.Normalize(vmin=min_sim, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# create a 1 row, 2 column matplotlib plot with the second column as a color bar and tiny
fig = plt.figure(figsize=(4.1*3, img.shape[0]/img.shape[1]*3))
gs = GridSpec(1, 5, width_ratios=[1, 1, 1, 1, 0.1])

# make last plot the colorbar
cax = fig.add_subplot(gs[:, 4])  # Span all rows in the last column
cb = fig.colorbar(scalarMap, cax=cax)
cb.locator = ticker.LinearLocator(3) # ticker.MaxNLocator(nbins=3)
cb.update_ticks()
cb.ax.set_yticklabels(['Dissimilar', 'Inbetween', 'Similar'])

# make first plot the image and we will add squares
img_ax = fig.add_subplot(gs[:, 0])  # Span all rows in the first column
img_ax.axis('off')
img_ax.imshow(img)

# for every sim, add the color to the image
min_sim = 100000
for i in range(len(all_encoings)):
    encoding = all_encoings[i]

    # compute cos similiarity
    sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)
    min_sim = min(min_sim, sim.item())

    # get the color
    color = scalarMap.to_rgba(sim.item())

    # get the goal location in pixels
    goal = all_goals[i]
    # note these goal locations may be adjusted by a pixel or two to fit the grid better
    goal = nearest_acceptable_grid_point(goal)
    x = goal[0] - 6
    y = goal[1] - 5
    width = 6
    height = 6

    # plot a rectangle at that location
    img_ax.add_patch(patches.Rectangle((x,y), width, height, color=color, alpha=1.0))


# add a star at the goal location
center_x, center_y = all_goals[goal_to_sim][0], all_goals[goal_to_sim][1]
img_ax.scatter(center_x - 3, center_y - 2, s=60, marker='*', color='gold', zorder=3)
img_ax.set_title("Function Encoder")
############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
# make second plot for function encoder ablation

# create the networks
reward_encoder = RewardEncoder(env_params, embed_dim) #.to("cuda:0")
reward_encoder.load_state_dict(torch.load(f'data/{save_dir_date_time_rea}/reward_encoder.pt', map_location="cpu"))

all_encoings = []
# Get encodings
for goal in all_goals:

    encoding = reward_encoder(torch.tensor(goal, dtype=torch.float32).unsqueeze(0) / 170.)

    all_encoings.append(encoding.cpu().detach())

# want to show cos sim relative to goal 0
goal_to_sim = 0

# make first plot the image and we will add squares
img_ax = fig.add_subplot(gs[:, 1])  # Span all rows in the first column
img_ax.axis('off')
img_ax.imshow(img)

# compute minimum of all cos similiarities
min_sim = 1
for i in range(len(all_encoings)):
    encoding = all_encoings[i]
    sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)
    min_sim = min(min_sim, sim.item())

# Create a color map. high is 1, low is 0.8
jet = plt.get_cmap(cmap_type)
cNorm = colors.Normalize(vmin=min_sim, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# for every sim, add the color to the image
for i in range(len(all_encoings)):
    encoding = all_encoings[i]

    # compute cos similiarity
    sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)
    min_sim = min(min_sim, sim.item())

    # get the color
    color = scalarMap.to_rgba(sim.item())

    # get the goal location in pixels
    goal = all_goals[i]
    # note these goal locations may be adjusted by a pixel or two to fit the grid better
    goal = nearest_acceptable_grid_point(goal)
    x = goal[0] - 6
    y = goal[1] - 5
    width = 6
    height = 6

    # plot a rectangle at that location
    img_ax.add_patch(patches.Rectangle((x,y), width, height, color=color, alpha=1.0))

# add a star at the goal location
center_x, center_y = all_goals[goal_to_sim][0], all_goals[goal_to_sim][1]
img_ax.scatter(center_x - 3, center_y - 2, s=60, marker='*', color='gold', zorder=3)
img_ax.set_title("Function Encoder Ablation")

############################################################################################################################
############################################################################################################################
############################################################################################################################
############################################################################################################################
# make third plot for FB
# create the networks
backward_network = BackwardMap(env_params, embed_dim)

# load backward from this code:
# torch.save([self.forward_network.state_dict(), self.backward_network.state_dict()], os.path.join(self.dir, 'model.pt'))
params = torch.load(f'data/{save_dir_date_time_fb}/model.pt', map_location="cpu")
backward_network.load_state_dict(params[1])

all_encoings = []
# Get encodings
for goal in all_goals:

    goal = torch.tensor(goal) / 170.
    encoding = backward_network(goal)
    all_encoings.append(encoding.cpu().detach())

# make first plot the image and we will add squares
img_ax = fig.add_subplot(gs[:, 2])  # Span all rows in the first column
img_ax.axis('off')
img_ax.imshow(img)

# compute minimum of all cos similiarities
min_sim = 1
for i in range(len(all_encoings)):
    encoding = all_encoings[i]
    sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)
    min_sim = min(min_sim, sim.item())

# Create a color map. high is 1, low is 0.8
jet = plt.get_cmap(cmap_type)
cNorm = colors.Normalize(vmin=min_sim, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# for every sim, add the color to the image
for i in range(len(all_encoings)):
    encoding = all_encoings[i]

    # compute cos similiarity
    sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)

    # get the color
    color = scalarMap.to_rgba(sim.item())

    # get the goal location in pixels
    goal = all_goals[i]
    # note these goal locations may be adjusted by a pixel or two to fit the grid better
    goal = nearest_acceptable_grid_point(goal)
    x = goal[0] - 6
    y = goal[1] - 5
    width = 6
    height = 6

    # plot a rectangle at that location
    img_ax.add_patch(patches.Rectangle((x,y), width, height, color=color, alpha=1.0))

# add a star at the goal location
center_x, center_y = all_goals[goal_to_sim][0], all_goals[goal_to_sim][1]
img_ax.scatter(center_x - 3, center_y - 2, s=60, marker='*', color='gold', zorder=3)
img_ax.set_title("Forward-Backward")

# make fourth plot for transformer
transformer = TransformerCritic(observation_space=env.observation_space['observation'].shape,
                                state_space=(2, ),
                                 action_space=(5,),
                                device="cpu")
transformer.load_state_dict(torch.load(f'data/{save_dir_date_time_transformer}/critic.pt', map_location="cpu"))
obs = _preproc_o(obs['observation'])

all_encoings = []
# Get encodings
for goal in all_goals:
    # predict rewards
    reward_function = get_reward_function(goal, 5)
    true_rewards = reward_function(torch.tensor(goal).unsqueeze(0)/170.)

    tensor_state = torch.tensor(goal).unsqueeze(0).unsqueeze(0)/170.
    encoding = transformer.get_latent_embedding(obs, tensor_state, true_rewards.unsqueeze(0))
    all_encoings.append(encoding.cpu().detach())


# make first plot the image and we will add squares
img_ax = fig.add_subplot(gs[:, 3])  # Span all rows in the first column
img_ax.axis('off')
img_ax.imshow(img)

# compute minimum of all cos similiarities
min_sim = 1
for i in range(len(all_encoings)):
    encoding = all_encoings[i]
    sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)
    min_sim = min(min_sim, sim.item())

# Create a color map. high is 1, low is 0.8
jet = plt.get_cmap(cmap_type)
cNorm = colors.Normalize(vmin=min_sim, vmax=1)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=jet)

# for every sim, add the color to the image
for i in range(len(all_encoings)):
    encoding = all_encoings[i]

    # compute cos similiarity
    sim = torch.nn.functional.cosine_similarity(encoding.flatten(), all_encoings[goal_to_sim].flatten(), dim=0)

    # get the color
    color = scalarMap.to_rgba(sim.item())

    # get the goal location in pixels
    goal = all_goals[i]
    # note these goal locations may be adjusted by a pixel or two to fit the grid better
    goal = nearest_acceptable_grid_point(goal)
    x = goal[0] - 6
    y = goal[1] - 5
    width = 6
    height = 6

    # plot a rectangle at that location
    img_ax.add_patch(patches.Rectangle((x,y), width, height, color=color, alpha=1.0))

# add a star at the goal location
center_x, center_y = all_goals[goal_to_sim][0], all_goals[goal_to_sim][1]
img_ax.scatter(center_x - 3, center_y - 2, s=60, marker='*', color='gold', zorder=3)
img_ax.set_title("Transformer")


# done
plt.tight_layout()
plt.savefig("cos_sim_pacman.png")

