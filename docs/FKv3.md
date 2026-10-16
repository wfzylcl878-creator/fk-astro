# FKv3 宿主处理方案

`normScheme=FKv3` 是新实例的默认选项，按静态恢复的 ML5 调用约定运行。枚举值为 `None=0/FKv2=2/FKv3=3`（FKv1=1 已移除，历史编号不复用，旧 FKv1 实例执行时报错需重选；FKv2 保留作旧实现对照，不能当成新路径的精度证明）。

目标是让后续自有模型的训练和推理使用同一套宿主约定。FKv3 选项要求图像端口为 1024×1024×3、输出为中心 512×512×3、参数恰好 6 个。CoreML 适配层在内部 CHW 缓冲与模型原生 CHW/HWC（可带单例 batch）之间按实际 strides 搬运；优先选择 `input/parameters/output` 命名端口。

FKv3 是本项目的宿主处理方案名，ML5 是它目前对应的官方模型版本；模型仍通过 Model path 独立选择。
界面入口为 Options → Processing scheme → FKv3。此前使用 `ML5` 枚举名的脚本须改用 `FKv3`；
参数 ID `normScheme`、数值 3、默认选择和算法保持不变。

## 归一化状态

先计算有效模式，再根据当前整图拟合一次状态。浮点样本保留为 double；整数样本先除以其最大采样值转为 double。

1. 四边各去 `int(尺寸×0.05)`，无效裁区回退整图。
2. 按比例重采样到长边 128，尺寸使用 llround，取点为 `floor(目标下标×源尺寸/目标尺寸)`，不加半像素偏移。小图可以放大取样。
3. 各通道所有正样本合并，排序；low 为位置 `(n−1)×0.001` 的线性插值分位数。high 固定为 1，来自 `arcsinh,wp=1`。
4. 若 `abs(high−low)<0.001`，重置 low=0、high=1。正样本经 `(p−low)×(1/(high−low))` 映射后，取下标 `n//2` 的上中位数 m。
5. 非 Lunar/Planetary 且 m>1e-6 时：`x=clamp(0.2/(m×(m+0.1)+1e-12),1,1e6)`；其余情况 x=1。
6. 保存 `a=asinh(x)`、`gain=log10(x/a)`。无正样本时 low=0、high=1、x=1，仍可继续运行。

像素和第六列共享该状态：

```text
Forward(p) = asinh((p−low)×(1/(high−low))×x)/a
params[5]  = float32(log10(x/a))
Inverse(y) = low + (sinh(y×a)/x)×(high−low)
```

拟合时过滤非正值，Forward 本身不裁掉负值，也不 clamp。越界块区域直接填模型域 0，只有有效图像区域执行 Forward；这与先补原图 0 再归一化不同。FKv3 忽略 Reflect border padding 控件，固定零填充。

## 六列及模式

| 下标 | 内容 |
|---|---|
| 0 | 有效 sharpenStars |
| 1 | 有效 adjustStarHalos |
| 2 | 有效 PSF Diameter × 0.5 |
| 3 | 有效 Automatic PSF，float 0/1 |
| 4 | 有效 sharpenNonstellar |
| 5 | 同一归一化状态的 log10gain |

Correct Only 将前五列置为 `[0,0,0,1,0]`，并令有效 lp=false；第六列仍由本图状态计算。

Lunar/Planetary 令有效 stars/halos=0、Automatic PSF=false，使用手动 PSF Diameter，nonstellar 仍有效；归一化 x=1，gain≈0.05483996856，Forward 仍为 arcsinh 曲线。界面按这些规则禁用控件，切换模式保留用户原先的设置。

## 分块和输出

设输出块 T=512、轴长 D、重叠比例 o：

```text
L = 2×floor(float32(T)×float32(o)/2)
n = 1 (D≤T)，否则 ceil(D/(T−L))
pitch = float32(D/n)
首原点=0，末原点=D−T
内部：a=float32(pitch×float32(k)), b=float32(T−pitch)
      origin=llround(double(a)−0.5×double(b))
小图 D<T：origin=(D−T)/2，保留有符号整数除法产生的负原点
```

接缝位于 `round(float32(pitch×k))`。长度 L 的余弦带居于接缝两侧，左带乘 `1−w`、右带乘 w；几何覆盖中超出该有效区的部分置零。仅有相邻块的一侧施加过渡，整图外边界不衰减。回写前先裁出 tile 与图像的交集，按负原点定位源像素。

输出在 double 模型域直接加权累加，不除以 wsum。整图融合后只执行一次 Inverse，最后 clamp 到 [0,1] 并转换为目标像素类型。整数图先乘最大采样值，再按默认 IEEE nearest/even 取整；这对应官方乘法后加 PCL 舍入常量的行为。高 overlap 下部分尺寸会有多条过渡带交叠，权重和可以略大于 1；实现保留该静态恢复行为。例如 D=513、o=0.5 时参考峰值约为 1.0045559170634588。

重叠区域写入由互斥锁保护。模型域全图累加器为 double，RGB 需要约 `W×H×3×8` 字节，单色为 `W×H×8` 字节；原图仍由 reader 按块读取。模块编译关闭 fast-math 和隐式乘加收缩，保留原点计算中的精度边界，已确认的 FMA 用 std::fma 表达。PCL 库保持原编译选项。

单色图从原始单通道拟合归一化，并将每个像素复制到三个模型输入通道。模型输出先按接缝窗口加权，再在每块内等权平均三个通道，累加到单通道全图，最后执行 Inverse。官方 sink 的延迟表达式先物化 `R+G`，再以 `addWeighted(R+G,1/3,B,1/3,0)` 求值；实现保留这一分组，使用 double 运算，不要求三个模型输出相同。具体设备的 SIMD/FMA 仍可能带来末位差异。

FKv3 对 CFA 使用同一普通图像通道映射：单通道马赛克作为灰度平面拟合、读取和写回，保留原坐标、图像通道数与 CFA 元数据，不先去马赛克或按颜色位置分开处理。这对应所追官方 PixInsight 执行链的静态行为，不能据此证明模型针对原始 CFA 训练或保证成图质量。旧方案的 CFA 和通用模型路径保留。

## 证据和验证范围

静态分析对象：`RCAstroSuite-pxm.dylib`，SHA256：

```text
48d7a4ce17bbd30238679de7879958618196aeb1a173dfb215c5c887c5b255ef
```

以下为 ARM64 IDA EA，不是 fat Mach-O 的原文件偏移：

| 行为 | 位置 |
|---|---|
| wp=1 配置解析和构造 | 内嵌 ML5 catalog；sub_43310 |
| 整图 Fit、gain 转 float 写入 | sub_F3348，0xF3560–0xF35C0 |
| 裁边、采样、分位数 | sub_4847C |
| 中位数、自适应 x、gain | sub_4811C；分母 FMA 在 0x481E8 |
| Forward / Inverse | sub_48F50 / sub_4911C |
| sinkIndex、sinkScale、有效 lp | sub_E11E4；0xE1CB0、0xE22A0 |
| 分块原点、接缝权重 | sub_F089C / sub_F1610；精度切换见 0xF0918–0xF0938 |
| 加权直接累加 | 0xF1D84 → sub_F2DC4 → sub_EA358；sub_ED078 返回前不除 wsum |
| 融合后整图逆变换 | sub_F3348，0xF3948 |
| 整数回写舍入 | sub_6BCC；8-bit 分支 0x6F58–0x6F68 先 FMUL 再 FADD 舍入常量 |
| 单色拟合读取、三通道输入复制 | sub_5920 / sub_666C；sub_5950 的 float32 分支 0x62F4–0x6300 |
| 单色每块平均、单通道累加 | sub_ED078 按源通道数分配全图；sub_EA358，0xEA3A8–0xEA430、0xEA5A4 |
| 平均表达式的分组和系数 | sub_21503C → sub_216D2C → sub_217B78 → sub_217008 → sub_1B2360（addWeighted） |
| CFA 使用普通图像读写路径 | sub_51D74 仅拒绝复数；sub_4D52C，0x4E068、0x4F018、0x4F11C、0x4F12C；sub_5950 / sub_6BCC 按颜色空间取 1/3 通道 |

离线验证运行：

```bash
python tests/run_offline.py
```

Windows 使用 Visual Studio C++ Build Tools，macOS/Linux 使用 clang++ 或 g++。只需要 Python 标准库，无模型文件。测试编译生产头文件，并直接提取生产 FKNormalizer、FitFKv3T、FKImageReaderT、WriteFKv3T 及张量元素读写函数，使用内存图像适配器执行。

覆盖 19 类：正样本为空、上中位数、最近邻坐标、通道汇总、分位插值、退化跨度、独立离线数值基准、模式/六列、正逆变换、浮点/整型精度、零填充、原点、接缝、融合顺序、不同三通道输出的逐块单色合并、原始 CFA 平面的同坐标复制、小图 RGB/单色、多块中心原样返回和张量 strides。CFA 探针执行普通单通道 reader；完整 PixInsight 的元数据及分支行为仍需运行验证。

这证明的是宿主算法和合成探针行为。完整 PCL/Objective-C++ 模块、CoreML 预测、设备误差、官方六阶段运行采集和实际成图精度尚未在本轮验证。跨平台 asinh/log10 及并发累加顺序仍可能产生末位差异。历史 FKv2 的成图表不适用于此 FKv3 路径。
