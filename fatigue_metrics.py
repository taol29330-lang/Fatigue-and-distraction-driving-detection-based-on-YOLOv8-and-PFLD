import numpy as np

# 下面这些索引来自你贴的代码；如果你的PFLD 106点定义不同，需要你改这里
IDX = {
    # left eye
    "le_top":  [41, 40, 42],
    "le_bot":  [36, 33, 37],
    "le_l":    35,
    "le_r":    39,
    # right eye
    "re_top":  [95, 94, 96],
    "re_bot":  [90, 87, 91],
    "re_l":    93,
    "re_r":    89,
    # mouth
    "m_top":   [63, 71, 67],
    "m_bot":   [56, 53, 59],
    "m_l":     52,
    "m_r":     61,
}

def eye_aspect_ratio(kps: np.ndarray) -> float:
    kps = kps.astype(np.float32)

    top = np.array([kps[i] for i in IDX["le_top"]])
    bot = np.array([kps[i] for i in IDX["le_bot"]])
    left_h = np.mean(bot[:, 1]) - np.mean(top[:, 1])
    left_w = kps[IDX["le_r"], 0] - kps[IDX["le_l"], 0]
    left_ear = left_h / (1e-6 + abs(left_w))

    top = np.array([kps[i] for i in IDX["re_top"]])
    bot = np.array([kps[i] for i in IDX["re_bot"]])
    right_h = np.mean(bot[:, 1]) - np.mean(top[:, 1])
    right_w = kps[IDX["re_r"], 0] - kps[IDX["re_l"], 0]
    right_ear = right_h / (1e-6 + abs(right_w))

    return float((left_ear + right_ear) / 2.0)

def mouth_aspect_ratio(kps: np.ndarray) -> float:
    kps = kps.astype(np.float32)
    top = np.array([kps[i] for i in IDX["m_top"]])
    bot = np.array([kps[i] for i in IDX["m_bot"]])
    h = np.mean(bot[:, 1]) - np.mean(top[:, 1])
    w = kps[IDX["m_l"], 0] - kps[IDX["m_r"], 0]
    mar = h / (1e-6 + abs(w))
    return float(mar)