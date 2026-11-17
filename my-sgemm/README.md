# my-sgemm

参考 `material` 中 Naive、Shared-memory tiling、WMMA 的思路重新组织的独立 CUDA mini 项目。整个目录可单独复制构建，不引用 `../material`，不下载 GoogleTest 或其他第三方源码。依赖仅为 CUDA Toolkit（含 cuBLAS）、CMake 和 C++ 编译器。已附带原仓库 `LICENSE`，分发时请保留。

## 结构与接口

```text
my-sgemm/
├── CMakeLists.txt
├── src/
│   ├── sgemm.h
│   ├── naive.cu
│   ├── tiled.cu
│   └── tensor_core.cu
├── bench/benchmark.cpp
├── profile/
│   ├── naive_report.txt
│   ├── tiled_report.txt
│   └── tensor_core_report.txt
└── README.md
```

统一计算行主序 `C[M,N] = A[M,K] * B[K,N]`，维度参数顺序为 M、N、K，支持正整数、矩形及尾块。只实现 alpha=1、beta=0。`src/sgemm.h` 为异步启动接口，调用方负责分配显存、检测错误及同步。

- Naive：16×16 线程块，一个线程计算一个输出。
- Tiled：16×16 shared-memory tile，越界补零；两次 block barrier 分别保护数据发布和覆盖。
- Tensor Core：一个 warp 计算一个 16×16 输出块，FP32 加载后在共享内存中转为 FP16，WMMA 用 FP32 累加，再经共享内存按边界写回。共享内存显式对齐，所有 warp lane 都参与 WMMA。没有尾块越界读写，也没有每次调用动态分配。

## 构建与运行

要求 CUDA Toolkit 11+、CMake 3.24+、兼容的 C++17 编译器。Windows 在 Visual Studio x64 开发者终端构建；Linux 使用 GCC。默认编译 sm_70，可自行指定目标架构（例如 86）。编译目标必须是支持 WMMA 的 sm_70 或更新架构；为本机显卡选择正确架构。

```sh
cd my-sgemm
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

当前机器检测到 RTX 4050 Laptop（sm_89），安装支持该显卡的 Toolkit 后可将上面的架构改为 `89`。

Linux 单配置生成器建议在 configure 时增加 `-DCMAKE_BUILD_TYPE=Release`。Linux 可执行文件为 `./build/sgemm_benchmark`；Windows 多配置生成器为 `./build/Release/sgemm_benchmark.exe`。

```sh
./build/sgemm_benchmark --m 1024 --n 1024 --k 1024 --csv results.csv
./build/sgemm_benchmark --m 37 --n 53 --k 29 --warmup 2 --repeat 5
./build/sgemm_benchmark --kernel tiled --m 2048 --n 2048 --k 2048
./build/sgemm_benchmark --help
```

`--kernel` 可选 all（默认）、naive、tiled、tensor_core、cublas。所有选择都会运行 cuBLAS 参考与计时。维度和迭代次数限制为 1 到 1000000，并检查设备 grid 限制；显存不足会报错退出。默认尺寸为 1024³，预热 5 次、测量 20 次，固定随机种子 42，输入均匀分布于 [-1,1]。GPU 默认使用设备 0。

## 计时与正确性

CUDA event 测量同一默认 stream 中重复调用的平均 GPU 时间。分配、主机传输、初始化和验证在计时区间外；WMMA 的共享内存加载和 FP16 转换包含在其 kernel 时间内。吞吐量为 `2*M*N*K/(ms*1e6)` GFLOP/s；speedup 为 cuBLAS 时间 / 当前实现时间。应在不运行 ncu 时采集性能表，避免 profiler replay 干扰。

cuBLAS 通过 `C^T=B^T*A^T` 适配行主序，使用 `CUBLAS_PEDANTIC_MATH` 作为完整 FP32 基线。所有实现使用同一组输入。Naive/Tiled 对照原始 FP32 基线；WMMA 对照“输入先量化为 FP16，再用 cuBLAS FP32 相乘”的参考，避免把输入量化损失混同内核错误。CSV 中 max_abs 和 relative_l2 始终相对原始 FP32 基线，failed_elements 使用各自验证参考；WMMA 通过验证不代表等价于完整 FP32 SGEMM。

逐元素判断 `abs(got-ref) <= 2e-5*K + 2e-4*abs(ref)`，拒绝 NaN/Inf；这是面向本项目有界随机输入的容差，不是任意输入的误差保证。验证前用 NaN 填充输出，检查遗漏写入。小于等于 200 万次乘加的形状额外用 CPU double oracle 检查 cuBLAS 布局和数值。测试覆盖对齐、非对齐、1×1、长 K 矩形和非法参数，任一数值检查失败返回非零。

## 性能对比表

当前开发环境未发现 nvcc、CMake，尚未编译或运行 GPU 测试。下面不填入参考项目的结果，`profile/*.txt` 也明确标记为待采集。运行上述命令后，将本机 CSV 结果填入表格，同时记录 GPU、CUDA/驱动版本、架构、尺寸及迭代次数。

| 实现 | 输入 / 累加 | 平均 ms | GFLOP/s | 相对 cuBLAS 加速比 |
| --- | --- | --- | --- | --- |
| Naive | FP32 / FP32 | 待测 | 待测 | 待测 |
| Tiled | FP32 / FP32 | 待测 | 待测 | 待测 |
| Tensor Core | FP16 / FP32 | 待测 | 待测 | 待测 |
| cuBLAS | FP32 / FP32 | 待测 | 待测 | 1.0 |

## Profiling 与瓶颈分析

三个 report 文件内提供可直接执行的 ncu 命令，运行后会被真实报告覆盖。需要安装 Nsight Compute 并具有 GPU 性能计数器访问权限。正则只选择目标内核，跳过其 5 次预热，采集下一次调用；不会把 cuBLAS 基线混入目标报告。更改 `--warmup` 时同步调整 `--launch-skip`。

预期机制与待验证假设如下，不能作为实测结论：

1. Naive 重复读取 A/B，缺少显式跨线程数据复用。理想化不考虑缓存的算术强度约为 0.25 FLOP/byte，而非参考注释中的 1。x 方向相邻线程读取连续 B 列，不能简单把 B 的访问称为不合并；实际事务还取决于 16×16 block 的 warp 布局。检查 Memory Workload、L1/L2 命中率和 DRAM 吞吐。
2. Tiled 将每个加载元素复用 16 次，理想化强度约为 4 FLOP/byte（忽略 C 写入和尾块）。但同步、共享内存指令、每线程单输出和较短计算链也有代价。检查 Scheduler、Warp State、Occupancy 与 shared-memory bank 指标，不能仅凭 padding 就断言消除了瓶颈。
3. WMMA 提高矩阵乘加吞吐，但一个 block 只有一个 warp，且不同输出 tile 会重复加载、转换输入。小矩阵可能由启动开销主导，大矩阵可能受转换、加载及并行调度限制。查看 Tensor 指令、SM 吞吐、寄存器和 occupancy，再判断是否需要多 warp tile、寄存器复用或预转换。

优化结论：本项目先提供清晰且统一的比较入口。shared-memory 复用和硬件矩阵指令是两条不同优化路径；FP16 有精度代价，不能仅用速度得出“更好的 SGEMM”。只有采集本机数据后才能确定三者的速度排序和实际瓶颈。下一步可逐次修改 tile 大小或 warp 布局，每次保留同形状正确性检查和无 profiler 的计时数据。
