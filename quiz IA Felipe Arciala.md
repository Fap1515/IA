# AI Engineering Scenarios 

> **Instructions:** For each scenario, analyze the plot(s), diagnose the issue, and propose concrete engineering actions.
> You may assume a standard supervised learning setup with train/validation splits.

---

## Scenario 1 — Generalization breakdown (possible overfitting)
![Scenario 1 — Training vs Validation Loss](scenario1_overfitting.png)

**Context:** You trained a model for a real business classification task (e.g., churn or fraud). You logged the training and validation loss across epochs.

### Questions
1. **Diagnose** what is happening around epoch ~25 (use the plot evidence).
    The plot indicates that the model improves rapidly in classifying both training and validation data during the first epochs. However, after epoch 25, the validation loss stops decreasing and starts to increase, while the training loss continues to go down. This behavior shows that the model is working well for the training data but is failing to generalize correctly to new data.
2. Is this **overfitting, underfitting, or neither**? Justify.
    This is a case of overfitting. Underfitting would cause poor performance on the training set, which is not what we see here. Overfitting, instead, results in great performance on training data as if the model had memorized it, but leads to poor generalization when it faces validation data.
    First, I would add early stopping. This would immediately help by stopping training at the point where the validation loss is minimal, which is typically when the model is generalizing best. It also reduces computational cost by avoiding extra epochs after performance starts to degrade.
    Second, to address the fact that overfitting is often linked to excessive model complexity, I would add L2 regularization (weight decay). This directly discourages overly large weights and helps reduce the train–validation gap.
    Finally, I would use dropout to reduce co-adaptation between neurons. By randomly turning off a fraction of units during training, the network is forced to learn more robust patterns, which usually improves generalization.
    F1 is the harmonic mean of precision and recall, so both metrics need to be considered together. I would request precision–recall plots as well as an F1 vs. epoch plot to understand how training improves the metric of interest. A high F1 can still be misleading if either precision or recall is too low individually, so examining both helps ensure the model is achieving a balanced performance.
5. Write a short plan: **what would you implement today** vs **what would you investigate next week**?
    What would you implement today:
    I would implement the strategies previously mentioned to obtain new, hopefully better results in the short term by reducing overfitting and co-adaptation. With these updates, it would be easier to determine whether the issue is related to the model itself or to the training data.
    What would you investigate next week:
    I would explore data-based strategies to further improve performance, regardless of whether clear data issues are found. For example, data augmentation could help increase variability by generating artificial samples from the original dataset. Additionally, I would run a hyperparameter sweep (adjusting parameters that are not learned during training, such as the learning rate, weight decay, and batch size) to identify configurations that lead to better performance and stronger generalization.
---

## Scenario 2 — Unstable optimization
![Scenario 2 — Oscillating Training Loss](scenario2_oscillating_loss.png)

**Context:** You are training a deep network using **vanilla SGD**. The training loss behaves as shown.

### Questions
1. What are the two most likely root causes (rank them)?
    The most likely root cause is an excessively high learning rate. If the step size is too large, the optimizer can overshoot the minimum and start “jumping” back and forth instead of converging, which leads to oscillations in the training loss.
The second likely cause is excessive mini-batch noise, which often occurs when the batch size is too small. In that case, each gradient estimate becomes noisy and the updates turn unstable.

2. What is the first change you would try? Explain why it is first.
    Reduce the learning rate. If that is not sufficient, I would then try increasing the batch size to obtain more stable gradient estimates. This directly targets the most likely cause and is a simple, quick adjustment to implement.
3. Would switching to Momentum / RMSProp / Adam help here? Explain the mechanism (not just “yes/no”).
    Momentum could help, but it remains sensitive to hyperparameters, especially the learning rate (α) and the momentum term (β), which controls how much inertia from past updates is retained versus how strongly the new gradient influences the direction.
    RMSProp and Adam address this issue more directly through adaptive rescaling. RMSProp divides the update by the square root of an exponential moving average of recent squared gradients, so when gradients are large the effective step size shrinks, helping reduce overshooting and dampen oscillations. Adam builds on this idea by also keeping an exponential moving average of the gradients (first moment) in addition to the squared gradients (second moment). This combines smoother update directions with adaptive step sizes, which typically results in more stable training than vanilla SGD in oscillatory regimes.
4. Suggest a diagnostic experiment that can distinguish between “bad LR” vs “bad batch size / noisy gradients”.
    Given that the learning rate is suspected to be too high, the first step would be to reduce it and observe whether the oscillations disappear. To confirm whether the issue is isolated or also influenced by batch size, a two-level factorial experiment (2^𝑘) varying both step size and batch size could then be conducted.
    By training the model across these combinations and comparing the loss curves, we can determine whether the instability is mainly caused by overshooting or by noisy gradients. The results would guide the selection of more appropriate hyperparameters or allow us to retain the best-performing pair if it already provides
5. What would you log (signals) to confirm the fix worked?
    Training and validation loss would be a strong starting point. If the loss–epoch curves stop oscillating and both decrease steadily as expected, we could reasonably assume that the instability has been addressed. A smoother training curve would further suggest that the optimizer is converging more reliably rather than overshooting the minimum.

---

## Scenario 3 — Backpropagation signal degradation (vanishing gradients)
![Scenario 3 — Gradient Magnitude Across Layers](scenario3_vanishing_gradients.png)

**Context:** You are training a 30-layer MLP. You log the average gradient magnitude per layer (from output layer backward).

### Questions

1. Diagnose what phenomenon the plot suggests and why it happens (use the chain rule argument).
   The phenomenon suggested by the plot is vanishing gradients. The issue is likely caused by how derivatives are computed during the backpropagation mechanism.
   Backpropagation first computes the derivative of the loss with respect to the output of the last layer (i.e., the network’s prediction). This is then multiplied by the derivative of the last layer’s output with respect to the previous layer’s output. Up to this point, the product is equivalent to the derivative of the loss with respect to the penultimate layer by the chain rule.
   The process repeats layer by layer until reaching the first one, where the derivative of the loss with respect to the parameters is finally obtained. Because many of these derivatives are typically smaller than 1, repeatedly multiplying them causes the gradient to shrink progressively, leading the early layers to learn very slowly or even stop learning.
2. Propose **three** model/architecture changes that directly target this (not optimizer-only suggestions).
   One alternative is to introduce residual connections into the network. In this setup, layers are arranged in blocks so that the block output changes from (y = F(x)) to (y = F(x) + x). During backpropagation, the derivative becomes (F'(x) + 1), where the +1 is critical to prevent the derivative from collapsing toward zero.
   Another change is to replace activation functions that compress the gradient, such as the sigmoid (whose derivative is upper-bounded by 0.25, its maximum value). Alternatives include ReLU (`max(0, x)`) or Leaky ReLU (`x` if `x > 0` and `αx` if `x < 0`), which reduces the risk of "dying" neurons by maintaining a small but nonzero derivative for negative values. In contrast, ReLU has a derivative of 1 when `x > 0` and 0 when `x < 0`.
   A third solution is to reduce the network depth. If a model can achieve the objective with fewer layers, there are fewer multiplicative terms in backpropagation, and therefore the resulting gradient is less severely diminished when each partial derivative is small.
3. Would changing the optimizer alone (e.g., Adam) solve it? Why/why not?
   No. Even with adaptive optimizers such as Adam, the ratio between the first moment and the square root of the second moment tends to stabilize within bounded values, typically on the order of 1, since both are moving averages of the gradient and its square. When gradients are consistently small, this ratio does not grow significantly, and the update size remains primarily determined by the learning rate. Numerically, Adam normalizes the scale of the updates but does not amplify weak gradients or compensate for the signal loss caused by vanishing gradients.
4. If you must keep the depth, what would you do with **initialization** and **activations**?
   First, activation functions can be replaced with non-saturating alternatives, as discussed previously. Additionally, weight initialization methods that approximately preserve the variance of activations across layers should be used to prevent the signal from fading or exploding as it propagates through the network. These methods are chosen according to the activation function employed. For example, He initialization compensates for the signal loss associated with ReLU by scaling the initial variance as `2 / fan_in`, where fan-in corresponds to the number of input connections to the neuron.
5. What evidence (additional plot/log) would you collect to confirm the diagnosis?
   I would collect a log table with the magnitude of gradients per layer at each epoch, for example by using the L2 norm of the weight gradients. This would allow a direct comparison of how the training signal evolves across the network. If the diagnosis is correct, the layers closer to the input should consistently exhibit smaller values than the deeper layers.

---

## Optional (bonus) — Short technical writing
Pick **one** scenario and write a short “incident report” (max 10 lines) including:
- Symptom
- Probable cause
- Immediate mitigation
- Longer-term fix
Scenario 2:
The training loss plot shows an oscillatory pattern, whereas a decreasing curve would be expected as prediction error is reduced through parameter optimization. The most likely cause is a poor hyperparameter choice, particularly an excessively large step size in mini-batch gradient descent, which can lead to overshooting. Although a very small batch size could introduce noise, the consistent oscillation makes this explanation less likely.

The immediate fix is to reduce the step size. A two-level factorial experiment (2^𝑘) varying step size and batch size could help determine whether one or both hyperparameters are responsible and guide the selection of better values.

As a longer-term solution, switching to adaptive optimizers such as RMSProp or Adam may help, since they dynamically scale updates using moving averages of past gradients, improving stability and convergence. The fix can be validated by recomputing the training and validation loss curves to confirm a smooth downward trend and good generalization.

**References**
Bishop, C. M., & Bishop, H. (2024). Deep Learning: Foundations and Concepts. Springer.

Bishop, C. M. (2006). Pattern Recognition and Machine Learning. Springer.

Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep Learning. MIT Press. http://www.deeplearningbook.org

Srivastava, N., Hinton, G., Krizhevsky, A., Sutskever, I., & Salakhutdinov, R. (2014). Dropout: A simple way to prevent neural networks from overfitting. Journal of Machine Learning Research, 15(1), 1929–1958. https://jmlr.org/papers/v15/srivastava14a.html

Martínez Vargas, J. D. (n.d.). ArtificialIntelligenceIM [Repositorio de GitHub]. GitHub.
https://github.com/jdmartinev/ArtificialIntelligenceIM

OpenAI. (2025). ChatGPT (GPT-5.2) [Research, Comprehension, and Writing Assistant]. https://chat.openai.com
