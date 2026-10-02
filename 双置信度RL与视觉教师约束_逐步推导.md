# 单一目标下的双置信度 RL 与视觉教师约束：从事件定义到联合策略梯度的逐步推导

**日期：** 2026-10-02  
**性质：** 研究设计与条件性数学推导；尚未运行模型训练，不包含效果结论。  
**目标：** 逐步解释为什么得到下面这个统一目标，以及每一项分别解决什么问题：

\[
\boxed{
J(\theta)=
\mathbb E\!\left[
2V+Y-(v-V)^2-V(r-Y)^2
\right]
-\beta\,
\mathbb E_{x,x^+}
D_{\mathrm{KL}}
\!\left(
\pi_\theta(\tau\mid x)
\Vert
\mu(\tau\mid x^+)
\right).
}
\]

本文重点解释推导，不把该目标表述为数学上唯一的选择。它是从一组明确要求反推得到的一种合法构造：

1. 视觉置信度应诚实报告“必要视觉事实正确”的概率；
2. 条件答案置信度应诚实报告“视觉事实正确时，最终答案正确”的概率；
3. 模型不能通过故意生成错误内容、再诚实地报告低置信度来获得高回报；
4. 学生还应利用清晰视觉教师提供的内容纠正；
5. 教师只约束内容，不直接规定学生应报告多高的置信度；
6. 所有信号在同一批 rollout、同一目标和同一次参数更新中结合。

---

# 0. 先看完整逻辑链

整套推导可以压缩成五步：

\[
\boxed{
\text{先定义两个概率事件}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{用平方评分得到诚实报告}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{加入正确性效用，防止“诚实地答错”}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{约束效用系数，避免提高视觉正确率反而降低回报}
}
\]

\[
\Downarrow
\]

\[
\boxed{
\text{加入内容策略到视觉教师的 KL 约束，并拉格朗日化}
}
\]

最终得到一个统一的 RL 目标：

- **终局奖励**评价视觉事实、答案和两个 verbal confidence；
- **教师 KL**提供逐 token 的内容纠正；
- **confidence 不模仿教师**，仍由学生当前输出的核验结果训练。

---

# 1. 随机变量、输出与信息状态

## 1.1 学生生成什么

给定学生可见输入

\[
x=(I,Q),
\]

其中 \(I\) 是受限、模糊或普通质量的视觉输入，\(Q\) 是问题。学生先生成内容：

\[
\tau=(Z,T,y),
\]

其中：

- \(Z\)：预先要求的必要视觉事实；
- \(T\)：推理文本；
- \(y\)：最终答案。

内容生成完以后，再生成两个口头置信度：

\[
c=(v,r).
\]

输出顺序是：

```text
Visual facts: Z
Reasoning: T
Answer: y
Visual confidence: v
Conditional answer confidence: r
```

将两个置信度放在内容之后，是为了避免置信度文本反过来改变已经被评分的视觉事实和答案。

---

## 1.2 两个二元核验事件

定义：

\[
V=
\begin{cases}
1,&\text{预先指定的必要视觉事实满足核验规则},\\
0,&\text{否则},
\end{cases}
\]

以及

\[
Y=
\begin{cases}
1,&\text{当前候选答案满足答案核验规则},\\
0,&\text{否则}.
\end{cases}
\]

注意：

- \(V\) 不是教师觉得“看起来合理”；
- \(Y\) 不是教师自己生成答案后的自评；
- 二者来自冻结、预先规定的核验协议；
- 必要视觉事实必须在生成前规定，不能让模型少写事实来逃避评分。

---

## 1.3 两个 verbal confidence 的概率语义

我们希望：

\[
v \approx P(V=1\mid H),
\]

以及

\[
r \approx P(Y=1\mid V=1,H).
\]

记：

\[
p=P(V=1\mid H),
\qquad
q=P(Y=1\mid V=1,H).
\]

因此：

\[
\boxed{v\text{ 的目标是 }p}
\]

\[
\boxed{r\text{ 的目标是 }q}
\]

这里把 \(r\) 称为 **conditional answer confidence（条件答案置信度）** 更准确。它不等于“整条推理链逻辑有效的概率”，因为核验标签仍然是最终答案事件 \(Y\)。

---

## 1.4 对信息状态 \(H\) 的严格说明

为了便于写式子，常把 \(H\) 写成当前输入与当前生成内容：

\[
H=(x,\tau).
\]

但严格地说，如果完整的 \(x,\tau\) 加上确定性核验规则已经唯一决定 \(V,Y\)，那么：

\[
P(V=1\mid H),\ P(Y=1\mid V=1,H)
\]

只能是 0 或 1。

非退化的 \(0.6,0.8\) 等概率，需要将 \(H\) 理解为：

- 报告器实际可利用的受限表示；
- 某一类相似信息状态；
- 隐含世界状态没有被完全观察时的条件信息；
- 或模型有限表达能力下的总体分布预测对象。

因此本文的 proper-scoring 推导是一个**条件分布层面的结论**，不是说单个完全确定样本天然包含一个客观的 0.73 标签。

---

# 2. 第一步：为什么视觉置信度使用平方评分

我们首先只看视觉事件 \(V\)。

给定信息状态 \(H\)，有：

\[
P(V=1\mid H)=p,
\qquad
P(V=0\mid H)=1-p.
\]

模型报告 \(v\in[0,1]\)。使用平方误差：

\[
(v-V)^2.
\]

其条件期望是：

\[
\begin{aligned}
\mathbb E[(v-V)^2\mid H]
&=
p(v-1)^2+(1-p)v^2\\
&=
p(v^2-2v+1)+(1-p)v^2\\
&=
pv^2-2pv+p+v^2-pv^2\\
&=
v^2-2pv+p.
\end{aligned}
\]

配方：

\[
v^2-2pv+p
=
(v-p)^2-p^2+p,
\]

所以：

\[
\boxed{
\mathbb E[(v-V)^2\mid H]
=
(v-p)^2+p(1-p).
}
\tag{1}
\]

其中：

\[
p(1-p)
\]

与模型报告 \(v\) 无关。因此，最小化期望平方误差等价于最小化：

\[
(v-p)^2.
\]

唯一最优值为：

\[
\boxed{v^*=p.}
\tag{2}
\]

这就是视觉置信度的 proper-reporting 性质。

---

## 2.1 为什么单次 \(V=1\) 不意味着目标 confidence 必须是 100%

训练时单个样本只给出：

\[
V\in\{0,1\}.
\]

如果当前样本 \(V=1\)，单次梯度当然会推动 \(v\) 变高；如果 \(V=0\)，则推动 \(v\) 变低。

但在所有具有相同信息状态 \(H\) 的样本上，若：

\[
P(V=1\mid H)=0.7,
\]

则 70% 样本为 1、30% 样本为 0。其期望平方损失的最优值正是：

\[
v=0.7.
\]

所以 proper scoring 不是把每个正样本强制标成“100%”，而是通过总体分布让最优报告等于条件正确率。

---

# 3. 第二步：为什么条件答案评分前面必须乘 \(V\)

现在看第二个报告：

\[
r \approx P(Y=1\mid V=1,H)=q.
\]

因为 \(r\) 的定义带有条件 \(V=1\)，所以只有视觉事实正确的样本，才对该条件事件提供直接监督。

因此使用：

\[
V(r-Y)^2.
\]

对它取条件期望：

\[
\begin{aligned}
\mathbb E[V(r-Y)^2\mid H]
&=
P(V=1\mid H)
\mathbb E[(r-Y)^2\mid V=1,H]\\
&=
p\,
\mathbb E[(r-Y)^2\mid V=1,H].
\end{aligned}
\]

在条件 \(V=1,H\) 下：

\[
P(Y=1\mid V=1,H)=q.
\]

因此：

\[
\begin{aligned}
\mathbb E[(r-Y)^2\mid V=1,H]
&=
q(r-1)^2+(1-q)r^2\\
&=
q(r^2-2r+1)+(1-q)r^2\\
&=
r^2-2qr+q\\
&=
(r-q)^2+q(1-q).
\end{aligned}
\]

所以：

\[
\boxed{
\mathbb E[V(r-Y)^2\mid H]
=
p(r-q)^2+pq(1-q).
}
\tag{3}
\]

当 \(p>0\) 时，唯一最优报告是：

\[
\boxed{r^*=q.}
\tag{4}
\]

---

## 3.1 为什么不直接使用 \((r-Y)^2\)

若不乘 \(V\)，则：

\[
\mathbb E[(r-Y)^2\mid H]
\]

的最优报告会是：

\[
r^*=P(Y=1\mid H),
\]

即无条件答案正确率，而不是：

\[
P(Y=1\mid V=1,H).
\]

因此 \(V\) 不是随意添加的 gate，而是由 \(r\) 的概率语义决定的。

---

## 3.2 \(p=0\) 时为什么无法学习 \(r\)

当：

\[
p=P(V=1\mid H)=0
\]

时，条件事件 \(V=1\) 从不发生，式 (3) 中与 \(r\) 有关的部分全部消失。

这意味着在该信息状态下：

\[
r
\]

不可识别。任何 \(r\) 都得到相同的期望评分。

这不是推导错误，而是条件概率本身的统计限制：如果条件事件没有样本，就无法学习该条件下的概率。

实践中必须报告：

- \(V=1\) 的样本量；
- 不同难度下 \(V=1\) 的覆盖率；
- \(r\) 的有效训练和评估样本数。

---

# 4. 第三步：为什么不能只用两项负 Brier

到目前为止，可以定义纯校准奖励：

\[
R_{\mathrm{cal}}
=
-(v-V)^2
-
V(r-Y)^2.
\tag{5}
\]

它可以让固定内容下的最优报告满足：

\[
v^*=p,\qquad r^*=q.
\]

但它没有鼓励模型提高内容正确率。

---

## 4.1 “诚实地答错”也可能获得高校准分

视觉平方评分在最优报告 \(v=p\) 下的不可约风险是：

\[
p(1-p).
\]

当：

\[
p=0
\]

时，模型总是视觉失败，但只要它诚实报告 \(v=0\)，视觉 Brier 风险为 0。

当 \(p=0\) 时：

\[
V(r-Y)^2=0
\]

也始终被关闭。

所以“总是视觉错误，并诚实地说自己不确定”可能获得很好的纯校准奖励。

同理，一个模型也可能保持低能力，但准确地预测自己的低能力。

因此：

\[
\boxed{
\text{proper calibration}
\not\Rightarrow
\text{high task capability}.
}
\]

必须另外增加正确性效用。

---

# 5. 第四步：加入视觉正确性和答案正确性效用

先考虑一般形式：

\[
\boxed{
R
=
aV+bY
-\rho(v-V)^2
-\sigma V(r-Y)^2,
}
\tag{6}
\]

其中：

\[
a,b,\rho,\sigma>0.
\]

四项分别表示：

- \(aV\)：奖励必要视觉事实正确；
- \(bY\)：奖励最终答案正确；
- \(\rho(v-V)^2\)：视觉置信度不诚实的惩罚；
- \(\sigma V(r-Y)^2\)：条件答案置信度不诚实的惩罚。

这里的系数是建模选择。数学可以告诉我们什么范围避免某些反向激励，但不能从纯数学中唯一决定所有任务偏好。

---

# 6. 为什么还要定义 \(g=P(Y=1\mid V=0,H)\)

答案可能在视觉事实错误时碰巧正确。例如：

- 视觉描述错了；
- 但模型借助语言先验猜中了答案。

因此定义：

\[
g=P(Y=1\mid V=0,H).
\tag{7}
\]

根据全概率公式：

\[
\begin{aligned}
P(Y=1\mid H)
&=
P(V=1\mid H)P(Y=1\mid V=1,H)\\
&\quad+
P(V=0\mid H)P(Y=1\mid V=0,H)\\
&=
pq+(1-p)g.
\end{aligned}
\tag{8}
\]

所以：

\[
\mathbb E[Y\mid H]
=
pq+(1-p)g.
\tag{9}
\]

\(g\) 只是分析变量，不需要模型额外报告第三个 confidence。

---

# 7. 将所有项逐项取期望

从式 (6) 出发：

\[
R
=
aV+bY
-\rho(v-V)^2
-\sigma V(r-Y)^2.
\]

逐项计算。

第一项：

\[
\mathbb E[aV\mid H]=ap.
\tag{10}
\]

第二项：

\[
\mathbb E[bY\mid H]
=
b[pq+(1-p)g].
\tag{11}
\]

第三项由式 (1)：

\[
\mathbb E[\rho(v-V)^2\mid H]
=
\rho(v-p)^2+\rho p(1-p).
\tag{12}
\]

第四项由式 (3)：

\[
\mathbb E[\sigma V(r-Y)^2\mid H]
=
\sigma p(r-q)^2+\sigma pq(1-q).
\tag{13}
\]

所以：

\[
\begin{aligned}
\mathbb E[R\mid H]
={}&
ap+b[pq+(1-p)g]\\
&-\rho(v-p)^2-\rho p(1-p)\\
&-\sigma p(r-q)^2-\sigma pq(1-q).
\end{aligned}
\tag{14}
\]

把与报告 \(v,r\) 有关的平方项单独放在后面：

\[
\boxed{
\mathbb E[R\mid H]
=
G(p,q,g)
-\rho(v-p)^2
-\sigma p(r-q)^2,
}
\tag{15}
\]

其中：

\[
\boxed{
G(p,q,g)
=
ap+b[pq+(1-p)g]
-\rho p(1-p)
-\sigma pq(1-q).
}
\tag{16}
\]

---

## 7.1 从式 (15) 立刻得到诚实报告

在固定 \(p,q,g\) 时：

\[
G(p,q,g)
\]

与报告值 \(v,r\) 无关。

由于 \(\rho>0\)：

\[
-\rho(v-p)^2
\]

在且仅在 \(v=p\) 时最大。

当 \(p>0,\sigma>0\) 时：

\[
-\sigma p(r-q)^2
\]

在且仅在 \(r=q\) 时最大。

所以：

\[
\boxed{
v^*=p,\qquad r^*=q\quad(p>0).
}
\tag{17}
\]

这说明加入正确性效用 \(aV+bY\) 没有破坏固定内容下的 proper-reporting 性质，因为正确性效用不依赖报告数值。

---

# 8. 诚实报告以后，奖励究竟鼓励什么

将：

\[
v=p,\qquad r=q
\]

代入式 (15)，得到最优报告下的价值：

\[
\boxed{
\mathbb E[R^*\mid H]
=
G(p,q,g).
}
\tag{18}
\]

展开式 (16)：

\[
\begin{aligned}
G
={}&
ap+bpq+b(1-p)g\\
&-\rho p+\rho p^2\\
&-\sigma pq+\sigma pq^2.
\end{aligned}
\]

合并：

\[
\boxed{
G
=
(a-\rho)p+\rho p^2
+p(b-\sigma)q
+\sigma pq^2
+b(1-p)g.
}
\tag{19}
\]

接下来检查：在其他概率固定时，提高 \(p,q,g\) 是否可能反而降低 \(G\)。

---

# 9. 对 \(q\) 的单调性：为什么要求 \(b\ge \sigma\)

对 \(q\) 求偏导：

\[
\boxed{
\frac{\partial G}{\partial q}
=
p(b-\sigma+2\sigma q).
}
\tag{20}
\]

因为：

\[
p\ge0,\quad q\ge0,\quad \sigma>0,
\]

若满足：

\[
\boxed{b\ge\sigma,}
\tag{21}
\]

则：

\[
b-\sigma+2\sigma q\ge0,
\]

于是：

\[
\frac{\partial G}{\partial q}\ge0.
\]

解释：

> 在视觉事实正确率 \(p\) 和其他条件保持不变时，提高“视觉正确条件下的答案正确率” \(q\)，不会降低最优期望奖励。

如果 \(b<\sigma\)，则在 \(q\) 较小时：

\[
b-\sigma+2\sigma q
\]

可能为负，意味着条件答案正确率略微提高，反而可能因为不可约 Brier 风险变化而降低总价值。

---

# 10. 对 \(p\) 的单调性：为什么要求 \(a\ge b+\rho\)

对 \(p\) 求偏导：

\[
\boxed{
\frac{\partial G}{\partial p}
=
a-bg-\rho+2\rho p
+(b-\sigma)q
+\sigma q^2.
}
\tag{22}
\]

在已经满足 \(b\ge\sigma\) 时：

\[
(b-\sigma)q\ge0,
\qquad
\sigma q^2\ge0,
\qquad
2\rho p\ge0.
\]

又因为：

\[
g\in[0,1],
\]

所以：

\[
-bg\ge-b.
\]

因此：

\[
\frac{\partial G}{\partial p}
\ge
a-b-\rho.
\tag{23}
\]

只要：

\[
\boxed{a\ge b+\rho,}
\tag{24}
\]

就有：

\[
\frac{\partial G}{\partial p}\ge0.
\]

解释：

> 在 \(q,g\) 保持不变时，提高必要视觉事实正确率 \(p\)，不会降低诚实报告后的最优期望奖励。

这是一个**全域充分条件**，不是必要条件。某些数据分布下，即使 \(a<b+\rho\)，导数也可能仍然为正。

---

# 11. 对 \(g\) 的单调性

对 \(g\) 求导：

\[
\boxed{
\frac{\partial G}{\partial g}
=
b(1-p)\ge0.
}
\tag{25}
\]

只要 \(b>0\)，在视觉错误的分支中提高答案正确率，也不会降低奖励。

这意味着当前效用仍然奖励“视觉事实错误但答案碰巧正确”。如果研究目标要求显式惩罚这种猜中，则需要修改效用，例如只奖励 \(VY\) 或增加 grounding consistency 项；那会形成不同的目标，不能继续直接使用本文全部推导。

---

# 12. 单位权重特例为什么得到系数 2

第一版取：

\[
b=\rho=\sigma=1.
\tag{26}
\]

一般充分条件变为：

\[
b\ge\sigma
\quad\Longrightarrow\quad
1\ge1,
\]

以及：

\[
a\ge b+\rho
\quad\Longrightarrow\quad
a\ge2.
\]

取最小简单值：

\[
a=2.
\]

得到：

\[
\boxed{
R
=
2V+Y
-(v-V)^2
-V(r-Y)^2.
}
\tag{27}
\]

此时：

\[
\boxed{
G(p,q,g)
=
p+p^2+pq^2+(1-p)g.
}
\tag{28}
\]

对应偏导：

\[
\frac{\partial G}{\partial p}
=
1+2p+q^2-g\ge0,
\tag{29}
\]

\[
\frac{\partial G}{\partial q}
=
2pq\ge0,
\tag{30}
\]

\[
\frac{\partial G}{\partial g}
=
1-p\ge0.
\tag{31}
\]

---

## 12.1 系数 2 的直观含义

在单位答案奖励、单位视觉 Brier 和单位条件答案 Brier 的设置下，提高 \(p\) 最坏时可能面临：

1. 从 \(V=0\) 的“碰巧答对”区域移到 \(V=1\) 区域，损失最多 1 单位答案效用；
2. 视觉 Brier 的不可约风险变化产生最多 1 单位的局部负影响。

因此给视觉正确性至少 2 单位效用，是覆盖全域最坏情况的一种保守充分做法。

但要强调：

\[
\boxed{
2\text{ 是充分值，不是普适最优值。}
}
\]

选择 \(2V+Y\) 同时意味着一种任务偏好：

| \(V\) | \(Y\) | 正确性效用 \(2V+Y\) |
|---:|---:|---:|
| 1 | 1 | 3 |
| 1 | 0 | 2 |
| 0 | 1 | 1 |
| 0 | 0 | 0 |

它认为：

\[
\text{视觉正确但答案错误}
>
\text{视觉错误但答案碰巧正确}.
\]

这适合强调 grounded reasoning 的任务，但不等价于“只最大化最终答案准确率”。

---

# 13. 数值例子：诚实报告为什么更优

假设某类信息状态满足：

\[
p=0.7,\qquad q=0.8,\qquad g=0.2.
\]

采用式 (27)。

诚实报告：

\[
v=0.7,\qquad r=0.8.
\]

诚实报告后的价值：

\[
\begin{aligned}
G
&=
p+p^2+pq^2+(1-p)g\\
&=
0.7+0.49+0.7\times0.64+0.3\times0.2\\
&=
1.698.
\end{aligned}
\]

若模型报告：

\[
v=0.9,\qquad r=0.6,
\]

则期望价值减少：

\[
(v-p)^2+p(r-q)^2
=
(0.9-0.7)^2+0.7(0.6-0.8)^2
=
0.04+0.028
=
0.068.
\]

所以：

\[
\mathbb E[R\mid H]
=
1.698-0.068
=
1.630.
\]

这展示了固定内容能力下，偏离真实概率会带来精确的二次期望损失。

---

# 14. 离散 verbal confidence 仍然怎样训练

实际模型生成的是百分比 token，例如：

\[
\mathcal C=\{0,0.01,\ldots,1.00\}.
\]

若报告只能取有限网格，连续最优值 \(p\) 不一定能精确表达。由平方项可知，最优报告是距离 \(p\) 最近的网格点。

最大量化误差为：

\[
\frac{1}{2M},
\]

其中网格间隔为 \(1/M\)。

如果报告策略随机地在多个网格值间分配概率，由于期望奖励对报告分布是线性的，最优随机策略会把全部概率质量放在最优网格点；若两点与 \(p\) 等距，则两者或其任意混合并列最优。

模型不需要对解析后的数字直接反向传播。它通过策略梯度提高高奖励 verbal-confidence 序列的生成概率。

---

# 15. 第五步：为什么还要加入视觉教师

奖励 \(R\) 能告诉模型：

- 视觉事实是否正确；
- 答案是否正确；
- 两个 confidence 是否诚实。

但终局奖励通常不能直接告诉模型：

> 某个具体内容 token 应该改成什么。

例如学生写“红色”，奖励只告诉它这条轨迹得分低；清晰视觉教师可以在该位置提高“蓝色”的概率，提供稠密 token-level correction。

因此加入一个训练时能看到更清晰视觉条件的冻结教师：

\[
x^+=(I^+,Q).
\]

教师内容策略记为：

\[
\mu(\tau\mid x^+).
\]

学生内容策略记为：

\[
\pi_\theta(\tau\mid x).
\]

教师只约束：

\[
\tau=(Z,T,y),
\]

不约束：

\[
c=(v,r).
\]

这样避免把教师的额外视觉信息优势直接变成学生的 confidence target。

---

# 16. 从约束优化得到 KL 正则项

希望学生最大化联合奖励，同时不要离清晰视觉教师太远：

\[
\begin{aligned}
\max_\theta\quad&
\mathbb E[R]\\
\text{s.t.}\quad&
\mathcal D(\theta)\le\delta,
\end{aligned}
\tag{32}
\]

其中：

\[
\boxed{
\mathcal D(\theta)
=
\mathbb E_{x,x^+}
D_{\mathrm{KL}}
\left(
\pi_\theta(\tau\mid x)
\Vert
\mu(\tau\mid x^+)
\right).
}
\tag{33}
\]

这是 student-to-teacher 的 reverse KL。

引入拉格朗日乘子：

\[
\beta\ge0.
\]

拉格朗日函数为：

\[
\begin{aligned}
\mathcal L(\theta,\beta)
&=
\mathbb E[R]
-\beta(\mathcal D(\theta)-\delta)\\
&=
\mathbb E[R]
-\beta\mathcal D(\theta)
+\beta\delta.
\end{aligned}
\tag{34}
\]

对固定 \(\beta\)，最后一项：

\[
\beta\delta
\]

与 \(\theta\) 无关。因此优化学生时可省略，得到：

\[
\boxed{
J(\theta)
=
\mathbb E[R]
-\beta\mathcal D(\theta).
}
\tag{35}
\]

代入式 (27)：

\[
\boxed{
J(\theta)=
\mathbb E\!\left[
2V+Y-(v-V)^2-V(r-Y)^2
\right]
-
\beta
\mathbb E
D_{\mathrm{KL}}
\left(
\pi_\theta(\tau\mid x)
\Vert
\mu(\tau\mid x^+)
\right).
}
\tag{36}
\]

---

## 16.1 固定 \(\beta\) 不等于严格满足 KL 预算

若只是手工固定 \(\beta\)，得到的是某个 reward–teacher trade-off，并不保证：

\[
\mathcal D(\theta)\le\delta.
\]

若真要执行原始约束，可进行对偶更新：

\[
\boxed{
\beta
\leftarrow
\left[
\beta+\eta_\beta(\mathcal D-\delta)
\right]_+.
}
\tag{37}
\]

当 KL 超预算时：

\[
\mathcal D>\delta,
\]

增大 \(\beta\)；反之减小。

神经网络非凸优化下，也不能由拉格朗日形式直接宣称强对偶或全局最优。

---

# 17. 为什么序列 KL 可以逐 token 计算

自回归学生策略：

\[
\pi_\theta(\tau\mid x)
=
\prod_{t=1}^{T}
\pi_\theta(\tau_t\mid x,\tau_{<t}).
\]

教师使用相同学生前缀：

\[
\mu(\tau\mid x^+)
=
\prod_{t=1}^{T}
\mu(\tau_t\mid x^+,\tau_{<t}).
\]

所以：

\[
\begin{aligned}
\log
\frac{\pi_\theta(\tau\mid x)}
{\mu(\tau\mid x^+)}
&=
\log
\prod_t
\frac{
\pi_\theta(\tau_t\mid x,\tau_{<t})
}{
\mu(\tau_t\mid x^+,\tau_{<t})
}\\
&=
\sum_t
\log
\frac{
\pi_\theta(\tau_t\mid x,\tau_{<t})
}{
\mu(\tau_t\mid x^+,\tau_{<t})
}.
\end{aligned}
\tag{38}
\]

定义：

\[
k_\theta(\tau;x,x^+)
=
\log
\frac{\pi_\theta(\tau\mid x)}
{\mu(\tau\mid x^+)}.
\tag{39}
\]

则：

\[
\mathbb E_{\tau\sim\pi_\theta}[k_\theta]
=
D_{\mathrm{KL}}(\pi_\theta\Vert\mu).
\tag{40}
\]

所以统一目标也可写成：

\[
\boxed{
J(\theta)
=
\mathbb E_{\tau,c\sim\text{Student}}
[
R-\beta k_\theta
].
}
\tag{41}
\]

教师不需要独立生成另一条完整轨迹；只需在学生实际前缀上给出每个内容 token 的 log-probability。

---

# 18. 统一目标的策略梯度如何逐步推导

## 18.1 联合生成概率分解

同一语言模型的联合概率写为：

\[
P_\theta(\tau,c\mid x)
=
\pi_\theta(\tau\mid x)
\kappa_\theta(c\mid x,\tau).
\tag{42}
\]

这里：

- \(\pi_\theta\)：内容段的自回归概率；
- \(\kappa_\theta\)：报告段的自回归概率；
- 二者共享同一套参数 \(\theta\)；
- 该分解不意味着两个独立网络或两个 head。

定义：

\[
U=2V+Y,
\tag{43}
\]

以及校准部分：

\[
S=-(v-V)^2-V(r-Y)^2.
\tag{44}
\]

所以：

\[
R=U+S.
\tag{45}
\]

统一目标：

\[
J
=
\mathbb E_{\tau,c}
[U+S-\beta k_\theta].
\tag{46}
\]

---

## 18.2 使用带显式参数项的 score-function 恒等式

对一般目标：

\[
\mathbb E_{z\sim p_\theta}[f_\theta(z)],
\]

有：

\[
\boxed{
\nabla_\theta
\mathbb E[f_\theta(z)]
=
\mathbb E[
f_\theta(z)\nabla_\theta\log p_\theta(z)
+
\nabla_\theta f_\theta(z)
].
}
\tag{47}
\]

这里：

\[
p_\theta(\tau,c)
=
\pi_\theta(\tau)\kappa_\theta(c\mid\tau),
\]

所以：

\[
\nabla\log p_\theta(\tau,c)
=
\nabla\log\pi_\theta(\tau)
+
\nabla\log\kappa_\theta(c\mid\tau).
\tag{48}
\]

而：

\[
f_\theta=R-\beta k_\theta.
\]

奖励 \(R\) 对参数没有额外显式导数；它通过采样动作依赖模型。\(k_\theta\) 显式包含 \(\log\pi_\theta\)，因此：

\[
\nabla f_\theta
=
-\beta\nabla k_\theta.
\]

教师固定，所以：

\[
\nabla k_\theta
=
\nabla\log\pi_\theta(\tau).
\tag{49}
\]

代入：

\[
\begin{aligned}
\nabla J
=
\mathbb E\big[
&(R-\beta k_\theta)
(\nabla\log\pi_\theta
+\nabla\log\kappa_\theta)\\
&-\beta\nabla\log\pi_\theta
\big].
\end{aligned}
\tag{50}
\]

---

## 18.3 为什么显式 KL 导数的最后一项消失

score-function 恒等式：

\[
\mathbb E_{\tau\sim\pi_\theta}
[\nabla\log\pi_\theta(\tau)]
=
\nabla\sum_\tau\pi_\theta(\tau)
=
\nabla 1
=
0.
\tag{51}
\]

所以：

\[
-\beta
\mathbb E[\nabla\log\pi_\theta]
=
0.
\]

式 (50) 化为：

\[
\nabla J
=
\mathbb E[
(R-\beta k_\theta)\nabla\log\pi_\theta
+
(R-\beta k_\theta)\nabla\log\kappa_\theta
].
\tag{52}
\]

---

## 18.4 为什么报告段不需要任务效用和教师 KL

给定固定内容 \(\tau\)，以下量不依赖之后采样的 confidence \(c\)：

\[
U=2V+Y,
\]

以及：

\[
k_\theta(\tau).
\]

条件 score 的期望为：

\[
\mathbb E_{c\sim\kappa_\theta}
[\nabla\log\kappa_\theta(c\mid\tau)\mid\tau]
=
0.
\tag{53}
\]

因此：

\[
\mathbb E[
U\nabla\log\kappa_\theta
\mid\tau
]
=
U\cdot0
=
0,
\]

以及：

\[
\mathbb E[
k_\theta\nabla\log\kappa_\theta
\mid\tau
]
=
k_\theta\cdot0
=
0.
\]

只有依赖 confidence 动作的校准项 \(S\) 保留下来：

\[
\boxed{
\nabla_\theta J
=
\mathbb E\left[
(R-\beta k_\theta)
\nabla_\theta\log\pi_\theta(\tau\mid x)
+
S
\nabla_\theta\log\kappa_\theta(c\mid x,\tau)
\right].
}
\tag{54}
\]

这就是统一目标对应的精确 score-function 梯度。

---

# 19. 式 (54) 实际意味着什么

## 19.1 内容 token 收到什么信号

内容段的 score：

\[
\nabla\log\pi_\theta(\tau\mid x)
\]

乘以：

\[
R-\beta k_\theta.
\]

所以内容 token 同时受到：

1. 视觉事实正确性 \(2V\)；
2. 最终答案正确性 \(Y\)；
3. 两个 confidence 的校准后果；
4. 与清晰视觉教师的 reverse-KL 约束。

因此内容策略不能把 calibration 当成与自己无关的“后处理任务”。它生成什么内容，会改变 \(V,Y\)，也会改变后续报告的评分。

---

## 19.2 confidence token 收到什么信号

报告段只需要：

\[
S=-(v-V)^2-V(r-Y)^2.
\]

原因是内容固定以后，confidence token 无法改变：

\[
2V+Y
\]

以及教师 KL。

因此给报告段减去这些与报告动作无关的常数，不改变期望梯度，却可以降低方差。

---

## 19.3 为什么仍是一次统一更新

虽然式 (54) 把内容 score 和报告 score 分开写，但：

- 两者属于同一个联合目标；
- 两者共享同一套参数；
- 两项梯度在同一次 backward 中累加；
- 最后只做一次 optimizer step。

因此这不是“先训内容，再训 confidence”的交替方法。

---

# 20. Baseline 如何加入而不改变期望梯度

可以给内容项加入不依赖当前内容动作的 baseline：

\[
b_{\pi}(x),
\]

得到：

\[
(R-\beta k_\theta-b_\pi(x))
\nabla\log\pi_\theta.
\]

也可给报告项加入给定内容后不依赖当前报告动作的 baseline：

\[
b_{\kappa}(H),
\]

得到：

\[
(S-b_\kappa(H))
\nabla\log\kappa_\theta.
\]

因为：

\[
\mathbb E[b_\pi\nabla\log\pi]=0,
\]

以及：

\[
\mathbb E[b_\kappa\nabla\log\kappa\mid H]=0,
\]

不会改变期望梯度。

若用同题其他 rollout 构造 leave-one-out baseline，必须保证 baseline 与当前被评分动作在相应条件下独立。

---

# 21. 如何把式 (54)落实成逐 token return

一种实现方式是：

- 在内容段每个 token 上累积教师代价：
  \[
  -\beta
  \log
  \frac{\pi_\theta(\tau_t\mid x,\tau_{<t})}
  {\mu(\tau_t\mid x^+,\tau_{<t})};
  \]
- 在内容结束后赋予终局联合奖励 \(R\)；
- 对内容 token 使用相应 return-to-go；
- 对 confidence token 使用校准回报 \(S\)。

必须避免两类重复计算：

1. 已经把 \(-\beta\log(\pi/\mu)\) 当作逐 token reward 后，又额外加入同一份直接 KL loss；
2. 给 confidence token 施加教师 KL，导致教师信息优势污染学生报告。

若使用 PPO/GRPO clipping、旧 rollout 多轮更新或组内标准差归一化，实际优化的是近似 surrogate，不能直接把它称为式 (54) 的无偏梯度。

第一版若要最大程度对齐推导，可使用当前策略采样、REINFORCE 或 leave-one-out baseline，并只更新一次。

---

# 22. 一次完整训练 step

下面给出与统一目标直接对应的一轮训练。

## Step 1：构造配对输入

学生输入：

\[
x=(I,Q).
\]

教师输入：

\[
x^+=(I^+,Q),
\]

其中 \(I^+\) 是清晰图、可靠裁图或额外视觉证据。

---

## Step 2：学生采样完整输出

从当前策略采样：

\[
(\tau,c)\sim
\pi_\theta(\tau\mid x)
\kappa_\theta(c\mid x,\tau).
\]

保存：

- 内容 token log-probability；
- confidence token log-probability；
- 内容边界；
- 解析后的 \(v,r\)。

---

## Step 3：冻结核验器给标签

根据预先固定协议得到：

\[
V\in\{0,1\},
\qquad
Y\in\{0,1\}.
\]

缺少必要视觉槽位不能从分母删除，应按事先规则判定失败或不可用。

---

## Step 4：计算终局奖励

\[
R
=
2V+Y-(v-V)^2-V(r-Y)^2.
\]

同时记录：

\[
S
=
-(v-V)^2-V(r-Y)^2.
\]

---

## Step 5：教师沿学生前缀评分

对每个内容位置 \(t\)，冻结教师计算：

\[
\log
\mu(\tau_t\mid x^+,\tau_{<t}).
\]

学生对应 log-probability 为：

\[
\log
\pi_\theta(\tau_t\mid x,\tau_{<t}).
\]

求和得到：

\[
k_\theta
=
\sum_{t\in\mathrm{content}}
\left[
\log\pi_\theta(\tau_t\mid x,\tau_{<t})
-
\log\mu(\tau_t\mid x^+,\tau_{<t})
\right].
\]

---

## Step 6：构造两个 segment 的策略梯度 loss

内容段可写成采样估计：

\[
\mathcal L_{\mathrm{content}}
=
-
\operatorname{sg}(R-\beta k_\theta-b_\pi)
\sum_{t\in\mathrm{content}}
\log\pi_\theta(\tau_t\mid x,\tau_{<t}).
\]

报告段：

\[
\mathcal L_{\mathrm{report}}
=
-
\operatorname{sg}(S-b_\kappa)
\sum_{t\in\mathrm{confidence}}
\log\kappa_\theta(c_t\mid x,\tau,c_{<t}).
\]

其中 \(\operatorname{sg}\) 表示把采样回报视为 score-function 权重，不对解析奖励直接反传。

若严格实现 reverse-KL 目标，需要正确处理 \(k_\theta\) 的显式参数依赖；使用“逐 token KL reward + score function”时不要再额外重复加入同一 KL。

---

## Step 7：一次更新

\[
\mathcal L
=
\mathcal L_{\mathrm{content}}
+
\mathcal L_{\mathrm{report}}.
\]

一次 backward，一次 optimizer step：

\[
\theta
\leftarrow
\theta-\eta\nabla_\theta\mathcal L.
\]

若采用对偶预算，再更新：

\[
\beta
\leftarrow
[\beta+\eta_\beta(\widehat{\mathcal D}-\delta)]_+.
\]

---

# 23. 与标准 OPSD forward KL 的区别

当前统一约束写的是：

\[
D_{\mathrm{KL}}
(
\pi_{\mathrm{Student}}
\Vert
\mu_{\mathrm{Teacher}}
),
\]

即 reverse KL。它可以写成学生 rollout 上的 sampled log-ratio，因此容易放入同一个 RL 目标。

标准 OPSD 常用的内容损失更接近：

\[
D_{\mathrm{KL}}
(
\mu_{\mathrm{Teacher}}
\Vert
\pi_{\mathrm{Student}}
),
\]

即 forward KL。在固定学生访问前缀上：

\[
\operatorname{KL}(q_t\Vert p_{\theta,t})
\]

的 logit 梯度为：

\[
p_{\theta,t}-q_t.
\]

两者不是同一个优化目标：

| 形式 | 方向 | 典型解释 |
|---|---|---|
| \(KL(\pi\Vert\mu)\) | Student → Teacher | 策略约束、mode-seeking、可写 sampled log-ratio |
| \(KL(\mu\Vert\pi)\) | Teacher → Student | 软标签交叉熵、覆盖教师概率质量 |

因此必须二选一地准确命名：

### 路线 A：严格保留本文统一约束

称为：

> **privileged reverse-KL policy constraint**

或：

> **OPD-style KL-regularized visual self-distillation**

### 路线 B：严格复现标准 OPSD

使用当前策略 rollout 的学生前缀，但将 forward KL 作为冻结访问分布上的辅助 supervised loss：

\[
\mathcal L_k
=
\mathcal L_{\mathrm{RL},k}
+
\lambda_D
\sum_t
KL(q_t\Vert p_{\theta,t}).
\]

它仍可在同一批 rollout、同一次更新中训练，但不再是式 (36) 那个 sequence reverse-KL 约束的精确同一推导。

论文中不能把两者无条件写成等价。

---

# 24. 与当前事实级主方案的关系

当前事实级方案通常为每个事实分别输出：

\[
c_1,\ldots,c_m,c_{\mathrm{answer}}.
\]

本文双置信度版本则压缩为：

\[
v=P(V=1\mid H),
\]

\[
r=P(Y=1\mid V=1,H).
\]

二者并不相同。

事实级版本的优点：

- 能看到每条事实的可靠性差异；
- 不会因一条事实错误就把全部视觉事件压成 \(V=0\)。

双置信度版本的优点：

- 更直接分离 perception 与 post-perception answer uncertainty；
- 数学上更容易写成条件概率；
- 输出更紧凑。

双置信度版本的风险：

- \(V\) 若定义为多个事实的 conjunction，会很稀疏；
- \(V=1\) 样本不足时，\(r\) 难以训练；
- 一条小事实错误与多条严重错误都可能得到同一个 \(V=0\)。

因此首轮实验应同时报告：

- 必要事实的数量；
- \(V=1\) 比例；
- 每事实正确率；
- \(r\) 的有效样本量。

---

# 25. 该推导真正证明了什么

本文证明的是以下条件性结论。

## 25.1 固定内容下的诚实报告

在正确事件定义、正评分系数和充分数据条件下：

\[
v^*=P(V=1\mid H),
\]

\[
r^*=P(Y=1\mid V=1,H).
\]

---

## 25.2 坐标单调性的充分条件

对一般奖励：

\[
aV+bY-\rho(v-V)^2-\sigma V(r-Y)^2,
\]

若：

\[
b\ge\sigma,
\qquad
a\ge b+\rho,
\]

则诚实报告后的价值对 \(p,q,g\) 分别非递减。

这是“其他坐标固定”的局部比较，不代表一次共享参数更新一定同时提高三个概率。

---

## 25.3 统一的教师约束目标

内容策略可在 KL 预算下最大化联合奖励，并通过拉格朗日形式得到：

\[
\mathbb E[R]-\beta KL.
\]

---

## 25.4 同一次更新的梯度分解

精确 score-function 梯度为：

\[
\mathbb E[
(R-\beta k)\nabla\log\pi
+
S\nabla\log\kappa
].
\]

这说明内容与报告可以在同一目标、同一批 rollout 和同一次参数更新中训练。

---

# 26. 该推导没有证明什么

它没有证明：

1. \(2V+Y\) 是唯一或最优任务效用；
2. 神经网络一定达到 Bayes-optimal confidence；
3. 共享参数可以同时提高 \(p,q,g\)；
4. 固定 \(\beta\) 一定满足 KL 预算 \(\delta\)；
5. 教师一定比学生正确；
6. clear-view Teacher 的内容分布一定只包含视觉纠正，而没有风格偏差；
7. PPO、GRPO、clipping 或组内标准化自动等价于精确梯度；
8. \(r\) 等于完整推理链逻辑正确率；
9. \(V=0\) 时的 \(r\) 可以被识别；
10. 实际训练会同时改善准确率、Brier 和 ECE。

这些都需要额外假设或实验。

---

# 27. 首轮实验必须验证的关键问题

建议按以下顺序。

## 27.1 奖励本身

比较：

1. 只有答案奖励 \(Y\)；
2. \(V+Y\)；
3. \(2V+Y\)；
4. \(2V+Y\) 加双 Brier；
5. 其他 \(a,b,\rho,\sigma\) 的合法系数。

报告：

- 视觉事实正确率；
- 最终答案准确率；
- \(V=1\) 比例；
- 两个 Brier/ECE；
- 高置信错误；
- 输出覆盖和格式失败。

---

## 27.2 Teacher 项

比较：

1. 无 Teacher；
2. reverse-KL 视觉教师约束；
3. 标准 OPSD forward KL；
4. Teacher 也蒸馏 confidence；
5. Teacher 只蒸馏内容。

由此区分：

- 收益来自额外计算；
- 收益来自 clear view；
- 收益来自 KL 方向；
- confidence mask 是否必要。

---

## 27.3 条件 confidence 的可识别性

必须报告：

\[
N_{V=1},
\]

即每个评估切片中 \(V=1\) 的样本数。

若 \(V=1\) 太少，可以考虑：

- 减少必要事实 conjunction 的数量；
- 将 \(V\) 改为固定事实正确比例；
- 使用每事实 confidence；
- 或增加一个无条件答案 confidence，而不是把全部答案 uncertainty 放入 \(r\)。

但这些会改变事件定义与推导，需要重新书写，而不能只改代码。

---

# 28. 最终总结

整个目标不是由一个公式突然出现，而是逐层反推得到：

\[
\boxed{
v=P(V=1\mid H),\
r=P(Y=1\mid V=1,H)
}
\]

决定了两个评分项：

\[
\boxed{
-(v-V)^2
-
V(r-Y)^2
}
\]

纯评分不能推动能力，因此加入：

\[
\boxed{
aV+bY
}
\]

要求诚实报告后的价值不因提高 \(p,q,g\) 而下降，得到一组充分条件：

\[
\boxed{
b\ge\sigma,\qquad
a\ge b+\rho
}
\]

单位设置：

\[
b=\rho=\sigma=1
\]

给出：

\[
\boxed{a=2}
\]

从而得到终局奖励：

\[
\boxed{
R
=
2V+Y-(v-V)^2-V(r-Y)^2.
}
\]

再加入“内容策略接近 clear-view Teacher”的约束：

\[
KL(\pi_\theta\Vert\mu)\le\delta,
\]

拉格朗日化后得到：

\[
\boxed{
J
=
\mathbb E[R]
-\beta KL(\pi_\theta\Vert\mu).
}
\]

最后利用联合自回归概率分解和 score-function 恒等式，得到一次统一更新的梯度：

\[
\boxed{
\nabla J
=
\mathbb E\left[
(R-\beta k)\nabla\log\pi
+
S\nabla\log\kappa
\right].
}
\]

其中：

- 内容 token 接收任务、校准后果和教师约束；
- confidence token 只接收与报告动作相关的 proper score；
- Teacher 不直接规定 confidence；
- 所有梯度在同一次 backward 中更新同一个模型。

这就是“单一目标下的双置信度 RL 与视觉教师约束”的完整推导链。
