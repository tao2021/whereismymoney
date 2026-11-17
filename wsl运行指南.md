# WSL 运行指南

本文说明如何在 Windows 编辑项目，通过 SSH 连接本机 WSL Ubuntu，编译和运行 CUDA 程序。

## 1. 本次使用的方式

代码保存在 Windows 中，Ubuntu 通过挂载路径直接访问同一份文件。没有复制代码，也没有使用同步工具。

| 环境 | 项目路径 |
| --- | --- |
| Windows | `C:\projects\my-sgemm` |
| WSL Ubuntu | `/mnt/c/projects/my-sgemm` |

WSL 中的 `/mnt/c/` 对应 Windows 的 C 盘。Windows 修改代码后，Ubuntu 可以直接读取；Ubuntu 在该路径下生成的 CSV 等文件，Windows 也能直接打开。这是文件系统挂载访问，不是两份文件之间的同步。

SSH 只负责提供远程执行命令的通道，不会自动传输或同步代码。本次连接的是同一台 Windows 电脑中的 WSL，因此能够使用上述共享路径。

## 2. 从 Windows 连接 Ubuntu

在 Windows PowerShell 中执行：

```powershell
ssh <SSH_USER>@<SSH_HOST>
```

将 `<SSH_USER>`、`<SSH_HOST>` 替换为自己的用户名和连接地址。本文路径为通用示例，需要按实际项目位置修改；连接前需要配置 WSL SSH 服务和访问端口。

如果已有免密认证，直接登录即可；否则在终端提示时输入密码。不要将密码写入命令、脚本或项目文件。

登录后执行的命令运行在 Ubuntu 中，使用 Ubuntu 安装的编译器、CUDA 和其他工具。

## 3. 检查环境并进入目录

在 Ubuntu SSH 终端中执行：

```bash
nvcc --version
cmake --version
g++ --version

cd /mnt/c/projects/my-sgemm
```

本次验证环境为 Ubuntu 24.04 WSL2、CUDA 12.0.76、GCC 12.4、CMake 3.28.3，GPU 为 RTX 4050 Laptop（sm_89）。

## 4. 首次配置、编译和测试

```bash
cmake -S . -B build-linux \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89

cmake --build build-linux -j4

ctest --test-dir build-linux --output-on-failure
```

参数含义：

- `-S .`：当前目录是源码目录。
- `-B build-linux`：配置及编译产物放到 `build-linux/`。
- `Release`：启用适合性能测试的发布构建。
- `89`：RTX 4050 的 CUDA 架构；换显卡时需要选择相应架构。
- `-j4`：最多使用 4 个并行构建任务。

本项目共有 5 项 CTest 测试。本次运行全部通过，其中非法参数测试按预期拒绝输入也算通过。

生成的 `build-linux/sgemm_benchmark` 是 Linux 可执行文件。即使它存放在 Windows C 盘对应的目录中，也应在 Linux 环境中运行。

## 5. 运行程序

默认运行 1024×1024 矩阵乘法，并导出结果：

```bash
./build-linux/sgemm_benchmark --csv results_1024.csv
```

运行更大的矩阵：

```bash
./build-linux/sgemm_benchmark \
  --m 2048 --n 2048 --k 2048 \
  --warmup 5 --repeat 50 \
  --csv results_2048.csv
```

查看参数帮助：

```bash
./build-linux/sgemm_benchmark --help
```

检查末尾是否显示 `Validation: PASS`，以及各实现的 `failed_elements` 是否为 0。`ms` 越小越快，`gflops` 越大越快；`speedup_vs_cublas` 大于 1 表示比本次测量的 cuBLAS 快。

CSV 保存在当前目录，Windows 中可从 `my-sgemm` 文件夹直接打开。

## 6. 修改代码后的日常操作

在 Windows 编辑并保存代码，然后在 Ubuntu 终端执行：

```bash
cd /mnt/c/projects/my-sgemm

cmake --build build-linux -j4 &&
ctest --test-dir build-linux --output-on-failure &&
./build-linux/sgemm_benchmark --csv results.csv
```

`&&` 表示上一条命令成功后才执行下一条。修改普通源代码后通常只需重新构建；更换架构或构建选项时应重新执行 CMake 配置。更换编译器或 CUDA Toolkit 时，建议使用新的构建目录。

也可以从 Windows PowerShell 用一条 SSH 命令完成：

```powershell
ssh <SSH_USER>@<SSH_HOST> 'cd /mnt/c/projects/my-sgemm && cmake --build build-linux -j4 && ctest --test-dir build-linux --output-on-failure && ./build-linux/sgemm_benchmark --csv results.csv'
```

执行结束后返回 Windows 终端。交互式 SSH 登录后，可以执行 `exit` 退出 Ubuntu 会话。

## 7. build-linux 和 build-wsl 的区别

| 目录 | 本项目中的用途 | 主要内容 |
| --- | --- | --- |
| `my-sgemm/build-linux/` | Ubuntu 编译目录 | CMake 缓存、目标文件、可执行程序和测试日志 |
| `my-sgemm/build-wsl/` | 本次 SSH 连接使用的辅助目录 | 记录服务器身份的 `known_hosts` |

`build-wsl/` 没有用于编译，这个名字只是当时人为选择的。WSL 运行的就是 Linux，保留 `build-linux/` 作为编译目录即可。

- 删除 `build-linux/` 不影响源码，但必须重新配置、编译后才能运行。
- 删除 `build-wsl/` 不影响编译和程序运行；依赖其中 `known_hosts` 的 SSH 命令需要重新记录或验证服务器身份。
- 普通 `ssh <SSH_USER>@<SSH_HOST>` 使用 SSH 默认配置，通常使用当前用户的 `~/.ssh/known_hosts`，不会自动读取项目中的 `build-wsl/known_hosts`。

Windows 和 Linux 的构建目录应分开，例如 `build-windows/` 与 `build-linux/`，避免混用编译缓存和可执行文件。

## 8. 如果连接的是另一台服务器

真正的远程服务器通常无法访问本机的 `/mnt/c/`。此时需要先传输代码，再在服务器上编译：

| 方式 | 适用情况 |
| --- | --- |
| `scp` | 偶尔复制文件或项目 |
| `rsync` | 经常更新，只传输变化的文件 |
| Git | 管理版本，在服务器拉取更新 |

SSH 连接本身不等于文件共享。只有远端已经挂载了相同存储时，才可以像本次 WSL 一样直接访问同一份源码。

## 9. 本次验证范围

项目已在上述 WSL 环境完成编译，5 项 CTest 全部通过，1024³ 和 2048³ 基准均通过数值验证。

尝试使用当前安装的 Nsight Compute 采集性能计数器时，工具报告 `Profiling is not supported on device 0`，未获得有效 profiling 数据。该问题不影响已完成的普通计时和正确性测试；失败日志位于 `my-sgemm/profile/naive_report.txt`。
