import numpy as np


class DualHeadLSTM:
    """Trainable NumPy LSTM with separate acceleration and covariance heads."""

    def __init__(
        self,
        hidden_size: int,
        a_max,
        max_log_scale: float = 2.0,
        seed: int = 0,
        *,
        input_size: int = 15,
        output_mode: str = "total_acceleration",
        loss_weights: dict[str, float] | None = None,
    ):
        self.input_size = int(input_size)
        self.output_mode = str(output_mode)
        self.feature_mean = np.zeros(self.input_size, dtype=float)
        self.feature_scale = np.ones(self.input_size, dtype=float)
        #隐藏层大小
        self.hidden_size = hidden_size
        #给加速度设置最大限制，神经网络可能输出过大的数值，但是现实应该是有限的扰动。
        self.a_max = np.asarray(a_max, dtype=float)
        #限制噪声变化范围，即最多100倍的变化。
        self.max_log_scale = float(max_log_scale)
        self.acceleration_error_covariance = np.zeros((3, 3), dtype=float)
        self.acceleration_output_gain = np.ones(3, dtype=float)
        self.residual_injection_gain = np.ones(3, dtype=float)
        self.loss_weights = {
            "acceleration": 1.0,
            "covariance": 0.7,
            "smoothness": 0.02,
            "acceleration_smoothness": 0.0,
        }
        if loss_weights is not None:
            self.loss_weights.update(loss_weights)
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.999
        self.adam_epsilon = 1e-8
        #生成可以复现的随机数
        rng = np.random.default_rng(seed)
        #input_scale是输入权重的标准差，recurrent_scale是循环权重的标准差。使用He初始化方法来初始化权重矩阵。
        input_scale = np.sqrt(2.0 / (self.input_size + hidden_size))
        recurrent_scale = 1.0 / np.sqrt(hidden_size)
        self.parameters = {
            #输入权重：输入15维，输出有4个门控，每个门控有hidden_size个神经元，所以权重矩阵的形状是(15, 4 * hidden_size)
            "wx": rng.normal(0.0, input_scale, (self.input_size, 4 * hidden_size)),
            #隐状态权重：链接上一时刻隐藏状态
            #循环权重矩阵，形状为(hidden_size, 4 * hidden_size)，用于将上一时刻的隐藏状态映射到当前时刻的门控。
            "wh": rng.normal(0.0, recurrent_scale, (hidden_size, 4 * hidden_size)),
            #偏置向量，形状为(4 * hidden_size,)，用于对门控进行偏置调整。
            "b": np.zeros(4 * hidden_size),
            #加速度头权重和偏置，形状分别为(hidden_size, 3)和(3,)，用于将隐藏状态映射到加速度输出。
            #对应ax,ay,az
            "wa": rng.normal(0.0, recurrent_scale, (hidden_size, 3)),
            "ba": np.zeros(3),
            #协方差头权重和偏置，形状分别为(hidden_size, 6)和(6,)，用于将隐藏状态映射到协方差输出。
            #对应logQ(3)和logR(3)
            "wc": rng.normal(0.0, recurrent_scale, (hidden_size, 6)),
            "bc": np.zeros(6),
        }
        #初始化遗忘门bias为1.0，让LSTM倾向于保留之前的记忆，对于时间序列，通常更稳定。
        self.parameters["b"][hidden_size:2 * hidden_size] = 1.0
        #Adam优化器的状态变量，用于存储梯度的一阶矩估计和二阶矩估计，以及步数计数器。
        self.adam_first = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.adam_second = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        self.adam_step = 0
    #前向传播，输入时间序列特征，输出预测的加速度和噪声协方差
    def forward(self, features, hidden=None, retain_cache=False):
        #输入(batch,T,15)，例如：20条轨迹、500步、15维特征。
        features = np.asarray(features, dtype=float)
        if features.ndim != 3 or features.shape[-1] != self.input_size:
            raise ValueError(
                f"features must have shape (batch, steps, {self.input_size})"
            )
        features = np.clip(
            (features - self.feature_mean) / self.feature_scale,
            -8.0,
            8.0,
        )
        batch, steps, _ = features.shape
        #初始化隐藏状态，ht短期输出，ct长期记忆
        hidden_states = np.zeros((batch, steps + 1, self.hidden_size), dtype=float)
        cell_states = np.zeros_like(hidden_states)
        if hidden is not None:
            hidden_states[:, 0] = hidden[0]
            cell_states[:, 0] = hidden[1]
        gates_i = np.zeros((batch, steps, self.hidden_size), dtype=float)
        gates_f = np.zeros_like(gates_i)
        gates_g = np.zeros_like(gates_i)
        gates_o = np.zeros_like(gates_i)
        #LSTM核心循环，输入xt、上一层的隐藏h_t-1，计算i、f、g、o四个门控，更新ct和ht。
        for index in range(steps):
            #数学：Z_t​=W_x*​x_t​+W_h*​h_t−1​+b，然后拆成i、f、g、o四个门控，分别计算各自的激活函数。
            preactivation = (
                features[:, index] @ self.parameters["wx"]
                + hidden_states[:, index] @ self.parameters["wh"]
                + self.parameters["b"]
            )
            i_raw, f_raw, g_raw, o_raw = np.split(preactivation, 4, axis=1)
            #输入门it​=σ(Wi​xt​+Ui​ht−1​)，决定写入多少新信息
            gates_i[:, index] = 1.0 / (1.0 + np.exp(-np.clip(i_raw, -30.0, 30.0)))
            #遗忘门ft​=σ(Wf​xt​+Uf​ht−1​)，决定保留多少旧信息
            gates_f[:, index] = 1.0 / (1.0 + np.exp(-np.clip(f_raw, -30.0, 30.0)))
            #候选记忆gt​=tanh(Wg​xt​+Ug​ht−1​)，生成新的候选记忆
            gates_g[:, index] = np.tanh(g_raw)
            #输出门ot​=σ(Wo​xt​+Uo​ht−1​)，决定输出多少信息
            gates_o[:, index] = 1.0 / (1.0 + np.exp(-np.clip(o_raw, -30.0, 30.0)))
            #更新LSTM记忆，数学公式：c_t​=f_t*​c_t−1​+i_t*​g_t​
            cell_states[:, index + 1] = (
                gates_f[:, index] * cell_states[:, index]
                + gates_i[:, index] * gates_g[:, index]
            )
            #隐藏状态：h_t​=o_t*​tanh(c_t​)
            hidden_states[:, index + 1] = gates_o[:, index] * np.tanh(cell_states[:, index + 1])
        #去掉第一个隐藏状态，得到每个时间步的隐藏状态
        encoded = hidden_states[:, 1:]
        #加速度输出和协方差输出，分别通过各自的权重和偏置进行线性变换，然后经过tanh激活函数限制输出范围。
        #加速度先输出[-1,1],再乘以最大加速度限制，得到加速度a
        acceleration_tanh = np.tanh(encoded @ self.parameters["wa"] + self.parameters["ba"])
        #协方差先输出[-1,1],再乘以最大噪声变化限制，得到logQ和logR
        covariance_tanh = np.tanh(encoded @ self.parameters["wc"] + self.parameters["bc"])
        acceleration = acceleration_tanh * self.a_max * self.acceleration_output_gain
        log_scales = covariance_tanh * self.max_log_scale
        next_hidden = (hidden_states[:, -1].copy(), cell_states[:, -1].copy())
        cache = None
        if retain_cache:
            cache = {
                "features": features,
                "hidden": hidden_states,
                "cell": cell_states,
                "i": gates_i,
                "f": gates_f,
                "g": gates_g,
                "o": gates_o,
                "acceleration_tanh": acceleration_tanh,
                "covariance_tanh": covariance_tanh,
            }
        return acceleration, log_scales, next_hidden, cache
    def evaluate_loss(self, features, labels):
        """Evaluate the supervised multi-head objective without updating weights."""

        acceleration, log_scales, _, _ = self.forward(features)
        target_acceleration = labels[..., :3]
        target_scales = labels[..., 3:]
        acceleration_error = (acceleration - target_acceleration) / self.a_max
        covariance_error = log_scales - target_scales
        acceleration_difference = (
            acceleration[:, 1:] - acceleration[:, :-1]
        ) / self.a_max
        scale_difference = log_scales[:, 1:] - log_scales[:, :-1]
        acceleration_loss = float(np.mean(np.square(acceleration_error)))
        covariance_loss = float(np.mean(np.square(covariance_error)))
        acceleration_smoothness_loss = (
            float(np.mean(np.square(acceleration_difference)))
            if acceleration_difference.size
            else 0.0
        )
        smoothness_loss = (
            float(np.mean(np.square(scale_difference))) if scale_difference.size else 0.0
        )
        total_loss = (
            self.loss_weights["acceleration"] * acceleration_loss
            + self.loss_weights["covariance"] * covariance_loss
            + self.loss_weights["smoothness"] * smoothness_loss
            + self.loss_weights["acceleration_smoothness"]
            * acceleration_smoothness_loss
        )
        return {
            "total": total_loss,
            "acceleration": acceleration_loss,
            "covariance": covariance_loss,
            "smoothness": smoothness_loss,
            "acceleration_smoothness": acceleration_smoothness_loss,
        }

    #训练函数 train_batch，这是反向传播部分。
    #输入features和labels，labels是之前build_dataset生成的，结构为Y=[ax​,ay​,az​,logQ,logR]
    def train_batch(self, features, labels, learning_rate, gradient_clip_norm=1.0):
        acceleration, log_scales, _, cache = self.forward(features, retain_cache=True)
        target_acceleration = labels[..., :3]
        target_scales = labels[..., 3:]
        acceleration_error = (acceleration - target_acceleration) / self.a_max
        covariance_error = log_scales - target_scales
        acceleration_difference = (
            acceleration[:, 1:] - acceleration[:, :-1]
        ) / self.a_max
        scale_difference = log_scales[:, 1:] - log_scales[:, :-1]
        #三个损失函数
        #加速度损失：La​=∣∣a−a^∣∣^2
        acceleration_loss = float(np.mean(np.square(acceleration_error)))
        #协方差损失：Lc​=∣∣logQ−logQ^∣∣^2+∣∣logR−logR^∣∣^2
        covariance_loss = float(np.mean(np.square(covariance_error)))
        acceleration_smoothness_loss = (
            float(np.mean(np.square(acceleration_difference)))
            if acceleration_difference.size
            else 0.0
        )
        #平滑（真实空间通常连续变化）损失：Ls​=∣∣s^k​−s^k−1​∣∣^2
        smoothness_loss = float(np.mean(np.square(scale_difference)))
        #最终损失函数：L=La​+0.7Lc​+0.02Ls​
        total_loss = (
            self.loss_weights["acceleration"] * acceleration_loss
            + self.loss_weights["covariance"] * covariance_loss
            + self.loss_weights["smoothness"] * smoothness_loss
            + self.loss_weights["acceleration_smoothness"]
            * acceleration_smoothness_loss
        )

        d_acceleration = (
            2.0
            * self.loss_weights["acceleration"]
            * acceleration_error
            / (acceleration_error.size * self.a_max)
        )
        if acceleration_difference.size:
            acceleration_smooth_gradient = (
                2.0
                * self.loss_weights["acceleration_smoothness"]
                * acceleration_difference
                / acceleration_difference.size
                / self.a_max
            )
            d_acceleration[:, 1:] += acceleration_smooth_gradient
            d_acceleration[:, :-1] -= acceleration_smooth_gradient
        d_log_scales = (
            2.0
            * self.loss_weights["covariance"]
            * covariance_error
            / covariance_error.size
        )
        if scale_difference.size:
            smooth_gradient = (
                2.0
                * self.loss_weights["smoothness"]
                * scale_difference
                / scale_difference.size
            )
            d_log_scales[:, 1:] += smooth_gradient
            d_log_scales[:, :-1] -= smooth_gradient

        d_acceleration_raw = (
            d_acceleration
            * self.a_max
            * self.acceleration_output_gain
            * (1.0 - cache["acceleration_tanh"] ** 2)
        )
        d_covariance_raw = d_log_scales * self.max_log_scale * (1.0 - cache["covariance_tanh"] ** 2)
        encoded = cache["hidden"][:, 1:]
        flat_hidden = encoded.reshape(-1, self.hidden_size)
        gradients = {name: np.zeros_like(value) for name, value in self.parameters.items()}
        gradients["wa"] = flat_hidden.T @ d_acceleration_raw.reshape(-1, 3)
        gradients["ba"] = d_acceleration_raw.sum(axis=(0, 1))
        gradients["wc"] = flat_hidden.T @ d_covariance_raw.reshape(-1, 6)
        gradients["bc"] = d_covariance_raw.sum(axis=(0, 1))
        d_hidden_heads = (
            d_acceleration_raw @ self.parameters["wa"].T
            + d_covariance_raw @ self.parameters["wc"].T
        )

        d_hidden_next = np.zeros((features.shape[0], self.hidden_size), dtype=float)
        d_cell_next = np.zeros_like(d_hidden_next)
        for index in range(features.shape[1] - 1, -1, -1):
            cell_current = cache["cell"][:, index + 1]
            cell_previous = cache["cell"][:, index]
            hidden_previous = cache["hidden"][:, index]
            gate_i = cache["i"][:, index]
            gate_f = cache["f"][:, index]
            gate_g = cache["g"][:, index]
            gate_o = cache["o"][:, index]
            tanh_cell = np.tanh(cell_current)
            d_hidden = d_hidden_heads[:, index] + d_hidden_next
            d_gate_o = d_hidden * tanh_cell
            d_cell = d_cell_next + d_hidden * gate_o * (1.0 - tanh_cell**2)
            d_gate_f = d_cell * cell_previous
            d_gate_i = d_cell * gate_g
            d_gate_g = d_cell * gate_i
            d_cell_next = d_cell * gate_f
            d_preactivation = np.concatenate(
                (
                    d_gate_i * gate_i * (1.0 - gate_i),
                    d_gate_f * gate_f * (1.0 - gate_f),
                    d_gate_g * (1.0 - gate_g**2),
                    d_gate_o * gate_o * (1.0 - gate_o),
                ),
                axis=1,
            )
            gradients["wx"] += cache["features"][:, index].T @ d_preactivation
            gradients["wh"] += hidden_previous.T @ d_preactivation
            gradients["b"] += d_preactivation.sum(axis=0)
            d_hidden_next = d_preactivation @ self.parameters["wh"].T

        gradient_norm = np.sqrt(sum(float(np.sum(value**2)) for value in gradients.values()))
        #梯度裁剪，防止梯度爆炸。
        if gradient_norm > gradient_clip_norm:
            scale = gradient_clip_norm / max(gradient_norm, 1e-12)
            gradients = {name: value * scale for name, value in gradients.items()}
        self.adam_step += 1
        for name in self.parameters:
            #Adam优化器的更新公式，使用一阶矩和二阶矩的估计来调整参数。
            self.adam_first[name] = (
                self.adam_beta1 * self.adam_first[name]
                + (1.0 - self.adam_beta1) * gradients[name]
            )
            self.adam_second[name] = (
                self.adam_beta2 * self.adam_second[name]
                + (1.0 - self.adam_beta2) * gradients[name] ** 2
            )
            first_hat = self.adam_first[name] / (1.0 - self.adam_beta1**self.adam_step)
            second_hat = self.adam_second[name] / (1.0 - self.adam_beta2**self.adam_step)
            self.parameters[name] -= learning_rate * first_hat / (
                np.sqrt(second_hat) + self.adam_epsilon
            )
        return total_loss
    #用于实际运行，输入当前一步的特征，输出下一步的加速度和噪声协方差，以及更新后的隐藏状态。
    def step(self, feature, hidden=None):
        feature = np.asarray(feature, dtype=float).reshape(self.input_size)
        feature = np.clip(
            (feature - self.feature_mean) / self.feature_scale,
            -8.0,
            8.0,
        )
        if hidden is None:
            hidden_previous = np.zeros(self.hidden_size, dtype=float)
            cell_previous = np.zeros(self.hidden_size, dtype=float)
        else:
            hidden_previous = np.asarray(hidden[0], dtype=float).reshape(-1, self.hidden_size)[0]
            cell_previous = np.asarray(hidden[1], dtype=float).reshape(-1, self.hidden_size)[0]

        preactivation = (
            feature @ self.parameters["wx"]
            + hidden_previous @ self.parameters["wh"]
            + self.parameters["b"]
        )
        i_raw, f_raw, g_raw, o_raw = np.split(preactivation, 4)
        gate_i = 1.0 / (1.0 + np.exp(-np.clip(i_raw, -30.0, 30.0)))
        gate_f = 1.0 / (1.0 + np.exp(-np.clip(f_raw, -30.0, 30.0)))
        gate_g = np.tanh(g_raw)
        gate_o = 1.0 / (1.0 + np.exp(-np.clip(o_raw, -30.0, 30.0)))
        cell_current = gate_f * cell_previous + gate_i * gate_g
        hidden_current = gate_o * np.tanh(cell_current)

        acceleration = np.tanh(
            hidden_current @ self.parameters["wa"] + self.parameters["ba"]
        ) * self.a_max * self.acceleration_output_gain
        log_scales = np.tanh(
            hidden_current @ self.parameters["wc"] + self.parameters["bc"]
        ) * self.max_log_scale
        hidden_next = (
            hidden_current.reshape(1, self.hidden_size),
            cell_current.reshape(1, self.hidden_size),
        )
        return acceleration, log_scales, hidden_next
