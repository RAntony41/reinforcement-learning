# Welcome to the introduction to reinforcement learning
This is my point of view to the reinforcement learning which I catagorize it in 2 main types **Value Base Method** and **Policy Gradient**.
Value Base Method is the method of using state and action table. Also, Value Base Method classify in to 2 types **Off-poilcy** and **On-poilcy**.

## Table of Contents
- [Introduction](#Introduction)
- [Value Base Method](#Value_Base_Method)
    - [Off-poilcy](#Off-poilcy)
    - [On-policy](#On-policy)
- [Policy Gradient](#Policy_Gradient)

# Introduction
**Agent** :decision-making entity

**Environment** :the world that agent interact with

**Exploration** : Agent try new action from the undiscover method
 
**Exploitation** : Agent use current knowledge to take action

### Policy
Function that determines the agent's behavior. It meant what action that a agent choose form state.
```math
\begin{align}
\pi(s|a)
\end{align}
```

### Return
The total discount reward the agent recives from a given state.
```math
\begin{align}
G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \cdots
\end{align}
```
```math
\begin{align}
G_t = r_t + \gamma G_{t+1}
\end{align}
```

### Model
The transition from the current state and action to the state and action.
```math
\begin{align}
p(s',r|s,a)
\end{align}
```

### State-value
Estimates the expected return from the state.
```math
\begin{align}
V_{\pi}(s) = \mathbb{E}_\pi [G_t|S_t = s]
\end{align}
```

### Action-value
Estimates the expected return from the state and action.
```math
\begin{align}
Q_{\pi}(s,a) = \mathbb{E}_\pi [G_t|S_t = s, A_t = a]
\end{align}
```

### Base-case scenario (Optimal)
```math
\begin{align}
V_*(s) \space and \space Q_*(s)
\end{align}
```

# Value_Base_Method
Most of them is create a table to map state and action. The challenge part is to update value to determine action from the state. So, their are many method to update the q-table.

**Bellman update equation**
```math
Q_*(s_t,a_t) = \mathbb{E}[r_t + \gamma \underset{a}{max} Q(s_{t+1},a)]
```

## Off-poilcy

**Q-learning**
```math
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_t + \gamma \underset{a}{max} Q(s_{t+1},a) - Q(s_t,a_t)]
```
**Monte Carlo**
```math
Q_\pi(s',a') \leftarrow Q_\pi(s',a') + \frac{1}{N(s,a)} [G_t - Q_\pi(s,a)]
```

## On-policy
**SARSA**
```math
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_t + \gamma Q(s_{t+1},a_{t+1}) - Q(s_t,a_t)]
```
**Expected SARSA**
```math
Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha [r_t + \gamma \sum \limits _{a'}\pi(a|s_{t+1}) Q(s_{t+1},a) - Q(s_t,a_t)]
```
# Policy_Gradient