import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
import random
from collections import deque

# ----- Hyperparameters -----
n_train_processes = 4
learning_rate = 0.0003
gamma = 0.99

update_interval = 20
max_train_steps = 80000
print_interval = update_interval * 100

buffer_limit = 500
replay_ratio = 4          # 한 번 rollout 후 replay update 횟수
replay_batch_size = 8     # rollout 단위 batch
truncation_clip = 10.0    # importance ratio clipping


class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 256)
        self.fc_pi = nn.Linear(256, 2)
        self.fc_q = nn.Linear(256, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        pi_logits = self.fc_pi(x)
        q = self.fc_q(x)
        return pi_logits, q

    def pi(self, x, softmax_dim=1):
        pi_logits, _ = self.forward(x)
        prob = F.softmax(pi_logits, dim=softmax_dim)
        return prob

    def q(self, x):
        _, q = self.forward(x)
        return q

    def v(self, x):
        prob = self.pi(x, softmax_dim=1)
        q = self.q(x)
        v = (prob * q).sum(dim=1, keepdim=True)
        return v


def worker(worker_id, master_end, worker_end):
    master_end.close()
    env = gym.make("CartPole-v1")
    obs, _ = env.reset(seed=worker_id)

    while True:
        cmd, data = worker_end.recv()

        if cmd == "step":
            obs, reward, terminated, truncated, info = env.step(int(data))
            done = terminated or truncated

            if done:
                obs, _ = env.reset()

            worker_end.send((obs, reward, done, info))

        elif cmd == "reset":
            obs, _ = env.reset()
            worker_end.send(obs)

        elif cmd == "close":
            env.close()
            worker_end.close()
            break

        else:
            raise NotImplementedError(f"Unknown command: {cmd}")


class ParallelEnv:
    def __init__(self, n_envs):
        self.nenvs = n_envs
        self.waiting = False
        self.closed = False
        self.workers = []

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends = master_ends
        self.worker_ends = worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker, args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(("step", int(action)))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return (
            np.stack(obs).astype(np.float32),
            np.array(rews, dtype=np.float32),
            np.array(dones, dtype=np.bool_),
            infos,
        )

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(("reset", None))
        return np.stack([master_end.recv() for master_end in self.master_ends]).astype(np.float32)

    def close(self):
        if self.closed:
            return

        if self.waiting:
            _ = [master_end.recv() for master_end in self.master_ends]

        for master_end in self.master_ends:
            master_end.send(("close", None))

        for worker in self.workers:
            worker.join()

        self.closed = True


class ReplayBuffer:
    def __init__(self, buffer_limit):
        self.buffer = deque(maxlen=buffer_limit)

    def put(self, rollout):
        self.buffer.append(rollout)

    def sample(self, n):
        n = min(n, len(self.buffer))
        return random.sample(self.buffer, n)

    def size(self):
        return len(self.buffer)


def make_rollout(model, envs, s):
    s_lst, a_lst, r_lst, mask_lst, mu_lst = [], [], [], [], []

    for _ in range(update_interval):
        with torch.no_grad():
            s_tensor = torch.from_numpy(s).float()
            prob = model.pi(s_tensor, softmax_dim=1)

        a = Categorical(prob).sample().numpy()
        s_prime, r, done, info = envs.step(a)

        s_lst.append(s.copy())
        a_lst.append(a.copy())
        r_lst.append((r / 100.0).copy())
        mask_lst.append((1.0 - done.astype(np.float32)).copy())
        mu_lst.append(prob.cpu().numpy().copy())

        s = s_prime

    rollout = {
        "s": np.array(s_lst, dtype=np.float32),         # [T, N, 4]
        "a": np.array(a_lst, dtype=np.int64),           # [T, N]
        "r": np.array(r_lst, dtype=np.float32),         # [T, N]
        "mask": np.array(mask_lst, dtype=np.float32),   # [T, N]
        "mu": np.array(mu_lst, dtype=np.float32),       # [T, N, A]
        "s_last": s.copy().astype(np.float32),          # [N, 4]
    }
    return rollout, s


def acer_update(model, optimizer, rollout):
    s = torch.tensor(rollout["s"], dtype=torch.float32)          # [T, N, 4]
    a = torch.tensor(rollout["a"], dtype=torch.long)             # [T, N]
    r = torch.tensor(rollout["r"], dtype=torch.float32)          # [T, N]
    mask = torch.tensor(rollout["mask"], dtype=torch.float32)    # [T, N]
    mu = torch.tensor(rollout["mu"], dtype=torch.float32)        # [T, N, A]
    s_last = torch.tensor(rollout["s_last"], dtype=torch.float32)# [N, 4]

    T, N, _ = s.shape
    action_dim = mu.shape[-1]

    with torch.no_grad():
        q_last = model.q(s_last)                                 # [N, A]
        pi_last = model.pi(s_last, softmax_dim=1)                # [N, A]
        q_ret = (pi_last * q_last).sum(dim=1)                    # [N]

    actor_loss_lst = []
    critic_loss_lst = []

    for t in reversed(range(T)):
        s_t = s[t]                                               # [N, 4]
        a_t = a[t].unsqueeze(1)                                  # [N, 1]
        r_t = r[t]                                               # [N]
        mask_t = mask[t]                                         # [N]
        mu_t = mu[t]                                             # [N, A]

        pi_t = model.pi(s_t, softmax_dim=1)                      # [N, A]
        q_t = model.q(s_t)                                       # [N, A]
        v_t = (pi_t * q_t).sum(dim=1)                            # [N]

        pi_a = pi_t.gather(1, a_t).squeeze(1)                    # [N]
        q_a = q_t.gather(1, a_t).squeeze(1)                      # [N]
        mu_a = mu_t.gather(1, a_t).squeeze(1)                    # [N]

        # Retrace target
        q_ret = r_t + gamma * q_ret * mask_t

        # importance ratio
        rho = pi_a.detach() / (mu_a + 1e-8)
        rho_bar = torch.clamp(rho, max=truncation_clip)

        # actor loss (truncated importance sampling)
        advantage = q_ret - v_t.detach()
        actor_main = -rho_bar * torch.log(pi_a + 1e-8) * advantage

        # bias correction
        rho_all = pi_t.detach() / (mu_t + 1e-8)
        correction_coeff = torch.clamp(1.0 - truncation_clip / (rho_all + 1e-8), min=0.0)

        q_adv = (q_t.detach() - v_t.detach().unsqueeze(1))
        actor_bias = -torch.sum(
            correction_coeff * pi_t * torch.log(pi_t + 1e-8) * q_adv,
            dim=1
        )

        actor_loss = actor_main + actor_bias

        # critic loss
        critic_loss = F.smooth_l1_loss(q_a, q_ret.detach(), reduction="none")

        actor_loss_lst.append(actor_loss.mean())
        critic_loss_lst.append(critic_loss.mean())

        # Retrace recursion
        retrace_coeff = torch.clamp(rho, max=1.0)
        q_ret = retrace_coeff * (q_ret - q_a.detach()) + v_t.detach()

    actor_loss = torch.stack(actor_loss_lst).mean()
    critic_loss = torch.stack(critic_loss_lst).mean()
    entropy = -(pi_t * torch.log(pi_t + 1e-8)).sum(dim=1).mean()

    loss = actor_loss + critic_loss - 0.01 * entropy

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
    optimizer.step()

    return loss.item(), actor_loss.item(), critic_loss.item()


def test(step_idx, model):
    env = gym.make("CartPole-v1")
    score = 0.0
    num_test = 10

    for _ in range(num_test):
        s, _ = env.reset()
        done = False

        while not done:
            with torch.no_grad():
                prob = model.pi(torch.from_numpy(s).float().unsqueeze(0), softmax_dim=1)
            a = prob.argmax(dim=1).item()

            s_prime, r, terminated, truncated, info = env.step(a)
            done = terminated or truncated
            s = s_prime
            score += r

    print(f"Step # : {step_idx}, avg score : {score / num_test:.1f}")
    env.close()


def main():
    envs = ParallelEnv(n_train_processes)

    model = ActorCritic()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    memory = ReplayBuffer(buffer_limit)

    step_idx = 0
    s = envs.reset()

    while step_idx < max_train_steps:
        rollout, s = make_rollout(model, envs, s)
        memory.put(rollout)

        # on-policy update
        loss, actor_loss, critic_loss = acer_update(model, optimizer, rollout)

        # off-policy replay update
        if memory.size() >= replay_batch_size:
            for _ in range(replay_ratio):
                batch = memory.sample(replay_batch_size)
                for sampled_rollout in batch:
                    acer_update(model, optimizer, sampled_rollout)

        step_idx += update_interval

        if step_idx % print_interval == 0:
            print(
                f"[TRAIN] step={step_idx}, "
                f"loss={loss:.4f}, actor={actor_loss:.4f}, critic={critic_loss:.4f}, "
                f"buffer={memory.size()}"
            )
            test(step_idx, model)

    envs.close()


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()