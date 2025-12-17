# Taichi Autodiff Limitations in MPM Solver

## Current Status

**Autodiff infrastructure已实现,但存在Taichi AD与MPM核心算法的兼容性问题。**

### 已完成 (✅)
1. **Loss Field创建** - loss_field with needs_grad=True (mpm_solver.py:45)
2. **Material Parameters标记** - ogden_mu, ogden_alpha, maxwell_G, maxwell_tau with needs_grad (mpm_solver.py:60-63)
3. **Particle State标记** - x, v, F, b_bar_e with needs_grad (fields.py:41-44)
4. **SPD Projection STE** - make_spd_ste() with straight-through estimator (decomp.py:115-126)
5. **Loss Kernels** - 四种可微loss: position/velocity/energy/com (autodiff_wrapper.py:45-72)
6. **Target Validation** - _validate_targets() prevents crashes (autodiff_wrapper.py:74-83)
7. **Unified STE Path** - 使用ti.static()统一STE/非STE路径 (mpm_solver.py:325-330)
8. **Kernel Separation** - reduce_energies分离为clear_energy_fields + reduce_energies (mpm_solver.py:360-368)

### 阻塞问题 (❌)

**P2G/G2P kernels包含Taichi autodiff不支持的操作:**

```python
# mpm_solver.py:96 - p2g()
for p in range(self.n_particles):
    # ... compute stress ...
    for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
        # ... scatter to grid ...
        if 0 <= grid_idx[0] < ...:  # ❌ Dynamic branching
            # ❌ Atomic scatter to grid (autodiff不支持)
            self.fields.grid_m[grid_idx] += weight * m_p
            self.fields.grid_v[grid_idx] += weight * (...)
```

**错误信息:**
```
[auto_diff.cpp:taichi::lang::ADTransform::visit@1099] Not supported.
```

**根本原因:**
- P2G中的grid scatter操作(atomic add到grid fields)
- 动态条件分支(bounds checking)
- G2P中的grid gather操作类似问题

**Taichi AD限制 (v1.7.4):**
- 不支持kernels内的atomic operations in AD tape
- 不支持数据依赖的动态分支
- P2G/G2P的scatter/gather pattern不在支持范围

## 可能的解决方案

### 方案1: 手动伴随方法 (Manual Adjoint)
**复杂度:** 高 (需要为p2g/g2p手写反向传播)
**优点:** 完整支持,准确梯度
**缺点:** 实现复杂,维护成本高,超出当前spec范围

### 方案2: 使用Taichi v1.8+ (如可用)
检查Taichi新版本是否改进了AD对atomic operations的支持。

### 方案3: 简化的可微模型
在autodiff模式下使用简化的物理模型(例如仅弹性,无网格散射),仅用于参数优化。
**缺点:** 不是真实MPM,梯度不准确。

### 方案4: 外部AD框架
使用JAX或PyTorch包装Taichi kernels,在外部构建计算图。
**优点:** 更强大的AD支持
**缺点:** 性能损失,架构变更大

## 推荐行动

1. **短期:** 在autodiff_wrapper.py中添加警告,说明当前autodiff仅支持loss对final state的梯度,无法反向传播到初始状态/材料参数(因为中间的P2G/G2P不可微)。

2. **中期:** 调研Taichi官方是否有MPM autodiff的参考实现或计划支持。

3. **长期:** 如需完整AD支持,考虑实现manual adjoint method或切换到支持更好的框架。

## 当前的实用价值

虽然P2G/G2P不可微,但已实现的部分仍有价值:
- ✅ Loss computation is differentiable
- ✅ Can compute gradient of loss w.r.t. final particle state
- ✅ Infrastructure ready for when/if Taichi improves AD support
- ⚠️ Cannot compute gradient w.r.t. material parameters/initial state through simulation

## 测试结果

```bash
$ conda run -n xengym python test_autodiff_step_by_step.py
Testing each kernel individually in Tape context...
  Testing clear_grid... OK
  Testing clear_particle_energy_increments... OK
  Testing clear_global_energy_step... OK
  Testing p2g... FAILED: Not supported.  # ← 阻塞在这里
```

## References

- Taichi Differentiable Programming: https://docs.taichi-lang.org/docs/differentiable_programming
- Known limitation: Atomic operations are not differentiable
- OpenSpec change: `update-mpm-autodiff-contact`
