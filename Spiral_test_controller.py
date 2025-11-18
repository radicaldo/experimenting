import pygame
import numpy as np
from dataclasses import dataclass

# ======================== MULTI-MODE SPIRAL SHOWCASE ========================
# Press the number keys listed in the HUD to swap between hand-tuned modes.
# Each mode tweaks the Stuart–Landau network to highlight a different look.
# ===========================================================================

N = 35
W, H = 1600, 900
DT = 0.08
C2 = 0.45        # non-isochrony, shared across presets
OMEGA0 = 1.8
OMEGA_GRAD = 0.25
NOISE_FREQ_SPREAD = 0.12

@dataclass
class SpiralMode:
    label: str
    title: str
    blurb: str
    mu: float
    K: float
    phase_lag: float
    noise: float
    radius: float
    kernel_sigma: float
    speed_base: float
    speed_gain: float
    speed_center: float
    speed_clip: tuple[float, float]
    speed_curve: str
    phase_bias_x: float
    phase_bias_y: float
    dot_base: float
    dot_gain: float
    dot_cap: float
    trail_len: int


def mode_list() -> list[SpiralMode]:
    return [
        SpiralMode(
            label="1",
            title="Lantern Bloom",
            blurb="Tight corkscrews that hug the dendrite like lantern vines.",
            mu=0.26,
            K=0.067,
            phase_lag=0.70,
            noise=0.09,
            radius=0.48,
            kernel_sigma=0.26,
            speed_base=0.032,
            speed_gain=0.028,
            speed_center=0.50,
            speed_clip=(-0.4, 0.8),
            speed_curve="linear",
            phase_bias_x=0.60,
            phase_bias_y=-0.35,
            dot_base=3.2,
            dot_gain=4.8,
            dot_cap=7.0,
            trail_len=130,
        ),
        SpiralMode(
            label="2",
            title="Storybook Arms",
            blurb="Looser arms that breathe outward before curling back in.",
            mu=0.32,
            K=0.085,
            phase_lag=0.50,
            noise=0.10,
            radius=0.50,
            kernel_sigma=0.27,
            speed_base=0.020,
            speed_gain=0.030,
            speed_center=0.48,
            speed_clip=(-0.3, 0.8),
            speed_curve="linear",
            phase_bias_x=0.55,
            phase_bias_y=-0.32,
            dot_base=2.8,
            dot_gain=4.2,
            dot_cap=6.0,
            trail_len=150,
        ),
        SpiralMode(
            label="3",
            title="Glow Pulse",
            blurb="Softer oscillations with rounded orbits and longer trails.",
            mu=0.30,
            K=0.075,
            phase_lag=0.58,
            noise=0.11,
            radius=0.46,
            kernel_sigma=0.24,
            speed_base=0.015,
            speed_gain=0.040,
            speed_center=0.45,
            speed_clip=(-0.35, 0.9),
            speed_curve="quadratic",
            phase_bias_x=0.52,
            phase_bias_y=-0.28,
            dot_base=3.0,
            dot_gain=4.5,
            dot_cap=6.5,
            trail_len=170,
        ),
    ]


PRESETS = mode_list()
current_mode_idx = 0

# These globals are updated whenever apply_mode is called.
mu = PRESETS[0].mu
K = PRESETS[0].K
phase_lag = PRESETS[0].phase_lag
noise = PRESETS[0].noise
radius = PRESETS[0].radius
kernel_sigma = PRESETS[0].kernel_sigma
speed_base = PRESETS[0].speed_base
speed_gain = PRESETS[0].speed_gain
speed_center = PRESETS[0].speed_center
speed_clip = PRESETS[0].speed_clip
speed_curve = PRESETS[0].speed_curve
phase_bias_x = PRESETS[0].phase_bias_x
phase_bias_y = PRESETS[0].phase_bias_y
trail_len = PRESETS[0].trail_len
dot_base = PRESETS[0].dot_base
dot_gain = PRESETS[0].dot_gain
dot_cap = PRESETS[0].dot_cap

# ======================== STATE ========================
pz = np.zeros(N, dtype=np.complex128)
pos = np.zeros((N, 2))
trails = [[] for _ in range(N)]
freq_noise = np.zeros(N)


# Fractal dendrite (same generator as other scripts)
def gen_dendrite(lv, start, ang, length, depth=0, segs=None, cols=None):
    if segs is None:
        segs = []
    if cols is None:
        cols = []
    if lv == 0:
        return segs, cols
    end = start + length * np.array([np.cos(ang), np.sin(ang)])
    segs.append((start.copy(), end.copy()))
    hue = depth / lv if lv else 0
    cols.append((
        int(255 * (0.5 + 0.5 * np.cos(6.28 * (hue + 0.0)))),
        int(255 * (0.5 + 0.5 * np.cos(6.28 * (hue + 0.33)))),
        int(255 * (0.5 + 0.5 * np.cos(6.28 * (hue + 0.67)))),
    ))
    new_len = length * 0.63
    gen_dendrite(lv - 1, end, ang - np.pi / 5.8, new_len, depth + 1, segs, cols)
    gen_dendrite(lv - 1, end, ang + np.pi / 5.8, new_len, depth + 1, segs, cols)
    return segs, cols


segs, cols = gen_dendrite(9, np.array([0.0, -1.15]), np.pi / 2, 0.95)
tips = np.array([s[1] for s in segs])
act = np.zeros(len(tips))


def reset_state(scale=1.0):
    global pz, pos, freq_noise, trails, act
    pz = scale * (np.random.randn(N) + 1j * np.random.randn(N))
    pos = np.random.uniform(-1, 1, (N, 2))
    freq_noise = NOISE_FREQ_SPREAD * np.random.randn(N)
    trails = [[] for _ in range(N)]
    act = np.zeros(len(tips))


def apply_mode(idx: int, reseed: bool = True):
    global current_mode_idx, mu, K, phase_lag, noise, radius, kernel_sigma
    global speed_base, speed_gain, speed_center, speed_clip, speed_curve
    global phase_bias_x, phase_bias_y, trail_len, dot_base, dot_gain, dot_cap

    current_mode_idx = idx % len(PRESETS)
    mode = PRESETS[current_mode_idx]
    mu = mode.mu
    K = mode.K
    phase_lag = mode.phase_lag
    noise = mode.noise
    radius = mode.radius
    kernel_sigma = mode.kernel_sigma
    speed_base = mode.speed_base
    speed_gain = mode.speed_gain
    speed_center = mode.speed_center
    speed_clip = mode.speed_clip
    speed_curve = mode.speed_curve
    phase_bias_x = mode.phase_bias_x
    phase_bias_y = mode.phase_bias_y
    trail_len = mode.trail_len
    dot_base = mode.dot_base
    dot_gain = mode.dot_gain
    dot_cap = mode.dot_cap
    if reseed:
        reset_state(scale=1.0)


apply_mode(0, reseed=True)


def w2s_left(p):
    return int((p[0] + 1.2) * (W // 2 - 50) / 2.4 + 25), int((1.2 - p[1]) * (H - 100) / 2.4 + 50)


def w2s_right(p):
    return int((p[0] + 1.6) * (W // 2 - 50) / 3.2 + W // 2 + 25), int((1.6 - p[1]) * (H - 100) / 3.2 + 50)


def intrinsic_freq(i):
    return OMEGA0 + OMEGA_GRAD * (0.8 * pos[i, 0] - 0.4 * pos[i, 1]) + freq_noise[i]


pygame.init()
screen = pygame.display.set_mode((W, H))
font = pygame.font.SysFont("consolas", 16)
hud_font = pygame.font.SysFont("consolas", 18)
clock = pygame.time.Clock()
frame = 0


running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        if e.type == pygame.KEYDOWN:
            if e.key == pygame.K_SPACE:
                reset_state(scale=0.6)
            elif e.key in (pygame.K_1, pygame.K_2, pygame.K_3):
                index = int(chr(e.key)) - 1
                apply_mode(index, reseed=True)
            elif e.key == pygame.K_TAB:
                apply_mode((current_mode_idx + 1) % len(PRESETS), reseed=True)

    # ======================== Stuart–Landau core ========================
    for i in range(N):
        coupling = 0j
        weight = 0.0
        for j in range(N):
            if i == j:
                continue
            dist = np.linalg.norm(pos[i] - pos[j])
            if dist < radius:
                w = np.exp(-(dist ** 2) / (2 * kernel_sigma ** 2))
                coupling += w * pz[j]
                weight += w
        if weight:
            coupling /= weight

        omega_i = intrinsic_freq(i)
        sl = ((mu + 1j * omega_i) - (1 + 1j * C2) * abs(pz[i]) ** 2) * pz[i]
        dzdt = sl + K * (np.exp(1j * phase_lag) * coupling - pz[i])
        dzdt += noise * (np.random.randn() + 1j * np.random.randn())
        pz[i] += dzdt * DT

        phase_dir = np.angle(pz[i]) + phase_bias_x * pos[i, 0] + phase_bias_y * pos[i, 1]
        amp = abs(pz[i])
        if speed_curve == "quadratic":
            amp_term = max(0.0, amp - speed_center)
            speed = speed_base + speed_gain * (amp_term ** 2)
        else:
            amp_delta = np.clip(amp - speed_center, speed_clip[0], speed_clip[1])
            speed = speed_base + speed_gain * amp_delta

        vel = speed * np.array([np.cos(phase_dir), np.sin(phase_dir)])
        pos[i] += vel * DT
        pos[i] = ((pos[i] + 1.2) % 2.4) - 1.2

        # tiny separation nudge
        sep = np.zeros(2)
        nsep = 0
        for j in range(N):
            if i == j:
                continue
            dvec = pos[i] - pos[j]
            d = np.linalg.norm(dvec)
            if 0.01 < d < 0.19:
                sep += dvec / d
                nsep += 1
        if nsep:
            pos[i] += sep / nsep * 0.008

        # dendrite touch
        for k, t in enumerate(tips):
            if np.linalg.norm(pos[i] - t) < 0.15:
                act[k] = min(1.0, act[k] + 0.38)

        trails[i].append(pos[i].copy())
        if len(trails[i]) > trail_len:
            trails[i].pop(0)

    act *= 0.94

    screen.fill((0, 0, 30))
    pygame.draw.line(screen, (50, 50, 100), (W // 2, 0), (W // 2, H), 2)

    # trails
    for i in range(N):
        for j in range(1, len(trails[i])):
            a = j / trail_len
            p1 = w2s_left(trails[i][j - 1])
            p2 = w2s_left(trails[i][j])
            pygame.draw.line(screen, (0, int(255 * a), int(180 * a)), p1, p2, 2)

    # agents
    for i in range(N):
        px, py = w2s_left(pos[i])
        amp = min(dot_cap, dot_base + dot_gain * abs(pz[i]))
        pygame.draw.circle(screen, (0, 255, 150), (px, py), int(amp))
        pygame.draw.circle(screen, (255, 255, 255), (px, py), int(amp), 1)

    # dendrite + glow
    for (s, e), c in zip(segs, cols):
        pygame.draw.line(screen, c, w2s_right(s), w2s_right(e), 2)
    for k, t in enumerate(tips):
        if act[k] > 0.05:
            px, py = w2s_right(t)
            intens = int(act[k] * 255)
            rad = int(4 + act[k] * 11)
            pygame.draw.circle(screen, (intens, intens // 3, 0), (px, py), rad)

    coh = np.abs(np.mean(pz / (abs(pz) + 1e-9)))
    txt = font.render(
        f"Mode [{PRESETS[current_mode_idx].label}] {PRESETS[current_mode_idx].title}  |  Coherence {coh:.3f}",
        True,
        (0, 255, 255),
    )
    screen.blit(txt, (10, H - 60))

    desc = PRESETS[current_mode_idx].blurb
    desc_text = font.render(desc, True, (180, 200, 255))
    screen.blit(desc_text, (10, H - 35))

    # Show preset menu on right panel
    menu_y = 20
    for idx, mode in enumerate(PRESETS):
        color = (255, 215, 0) if idx == current_mode_idx else (120, 140, 200)
        menu = hud_font.render(f"[{mode.label}] {mode.title}", True, color)
        screen.blit(menu, (W // 2 + 30, menu_y))
        menu_y += 24
        sub = font.render(mode.blurb, True, (80, 120, 200))
        screen.blit(sub, (W // 2 + 40, menu_y))
        menu_y += 28

    hint = font.render("TAB = next mode, SPACE = shuffle same mode", True, (150, 160, 200))
    screen.blit(hint, (10, 10))

    pygame.display.flip()
    clock.tick(60)
    frame += 1

pygame.quit()
