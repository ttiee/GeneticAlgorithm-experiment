# GA + charts (no seaborn, single-plot per figure, default colors)
import numpy as np
import random
import matplotlib.pyplot as plt

# --------- GA core ---------
GENE_LEN = 5  # 0..31

def fitness_x(x:int)->int:
    return x**5 - 1000

def encode(x:int):
    return np.array([(x>>i)&1 for i in range(GENE_LEN)][::-1], dtype=np.int8)

def decode(bits:np.ndarray)->int:
    v=0
    for b in bits:
        v=(v<<1)|int(b)
    return v

def init_pop(n:int):
    return np.random.randint(0,2,size=(n,GENE_LEN), dtype=np.int8)

def tournament_select(pop, fits, k=3):
    idxs = np.random.randint(0, len(pop), size=k)
    best = idxs[np.argmax(fits[idxs])]
    return pop[best].copy()

def one_point_cx(p1,p2, pc=0.9):
    if random.random() > pc: return p1.copy(), p2.copy()
    cx = random.randint(1, GENE_LEN-1)
    c1 = np.concatenate([p1[:cx], p2[cx:]])
    c2 = np.concatenate([p2[:cx], p1[cx:]])
    return c1,c2

def bitflip_mut(ind, pm=0.02):
    mask = np.random.rand(GENE_LEN) < pm
    ind[mask] = 1 - ind[mask]
    return ind

def clamp_domain(ind):
    x = decode(ind)
    x = min(31, max(0,x))
    return encode(x)

def evolve(pop_size=40, gens=52, pc=0.9, pm=0.02, elite=1, seed=42):
    random.seed(seed); np.random.seed(seed)
    pop = init_pop(pop_size)
    best_hist, avg_hist = [], []
    best_ind, best_fit = None, -np.inf

    for g in range(gens):
        xs = np.array([decode(ind) for ind in pop])
        fits = np.array([fitness_x(int(x)) for x in xs])

        avg_hist.append(fits.mean())
        gi = fits.argmax()
        if fits[gi] > best_fit:
            best_fit, best_ind = fits[gi], pop[gi].copy()
        best_hist.append(best_fit)

        # next generation
        new_pop = []
        elite_idx = np.argsort(-fits)[:elite]
        for ei in elite_idx:
            new_pop.append(pop[ei].copy())
        while len(new_pop) < pop_size:
            p1 = tournament_select(pop, fits, k=3)
            p2 = tournament_select(pop, fits, k=3)
            c1, c2 = one_point_cx(p1,p2,pc)
            c1 = clamp_domain(bitflip_mut(c1, pm))
            c2 = clamp_domain(bitflip_mut(c2, pm))
            new_pop.extend([c1,c2])
        pop = np.array(new_pop[:pop_size], dtype=np.int8)
    bx = decode(best_ind)
    by = fitness_x(bx)
    return bx, by, np.array(best_hist, dtype=float), np.array(avg_hist, dtype=float)

best_x, best_y, best_hist, avg_hist = evolve()

# --------- Charts (single-plot per figure) ---------

# 1) Fitness evolution (best & average)
plt.figure(figsize=(10,6))
plt.plot(best_hist, marker='o', label='最佳适应度')
plt.plot(avg_hist, marker='s', label='平均适应度')
plt.title('遗传算法进化过程')
plt.xlabel('迭代次数')
plt.ylabel('适应度值')
plt.legend()
plt.tight_layout()
plt.savefig('evolution_process.png', dpi=180)
plt.close()

# 2) Best fitness stair-step
plt.figure(figsize=(10,6))
plt.step(range(len(best_hist)), best_hist, where='post')
plt.title('最佳适应度变化')
plt.xlabel('迭代次数')
plt.ylabel('最佳适应度')
plt.tight_layout()
plt.savefig('best_fitness_step.png', dpi=180)
plt.close()

# 3) Convergence speed (delta of best fitness)
delta = np.diff(best_hist, prepend=best_hist[0])
plt.figure(figsize=(10,6))
plt.plot(delta, marker='^')
plt.axhline(0, linestyle='--')
plt.title('收敛速度分析（相邻代最佳适应度增量）')
plt.xlabel('迭代次数')
plt.ylabel('适应度改进量')
plt.tight_layout()
plt.savefig('convergence_speed.png', dpi=180)
plt.close()

# 4) Objective curve & optimal point
xs = np.arange(0, 32)
ys = xs**5 - 1000
plt.figure(figsize=(10,6))
plt.plot(xs, ys, marker='o', label='目标函数 y = x^5 - 1000')
plt.scatter([best_x],[best_y], s=120, label=f'遗传算法最优解: x={best_x}, y={best_y:.2f}')
# Theoretical best is x=31
th_x, th_y = 31, 31**5 - 1000
plt.scatter([th_x],[th_y], s=120, marker='^', label=f'理论最优解: x={th_x}, y={th_y:.2f}')
plt.title('目标函数与最优解')
plt.xlabel('x 值')
plt.ylabel('y 值')
plt.legend()
plt.tight_layout()
plt.savefig('objective_function.png', dpi=180)
plt.close()


