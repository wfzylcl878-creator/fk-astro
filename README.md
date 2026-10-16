# fk-astro

PixInsight 里的 CoreML 分块推理模块(`.mlpackage` / `.mlmodel` 明文直读),
默认提供 **FKv3 宿主处理方案**：整图 arcsinh 归一化、六列参数、带上下文的分块、
接缝融合后整图逆变换。逻辑依据官方 ML5 宿主的 dylib 静态分析恢复，当前完成离线验证，
尚未验证新路径的官方成图误差。FKv2 保留作旧实现对照(FKv1 已移除)。

两用:

1. **通用骨架** — 在 PixInsight 里跑你自己训练的去噪/锐化类 CoreML 模型;
2. **BXT 复现验证** — 选择 FKv3 方案,与官方
   BlurXTerminator 2.6.9 输出做逐像素对比。

## 目录结构

```
fk-astro/
├── Makefile              构建脚本(双架构通用二进制)
├── install.sh            签名 + 拷贝 + 启动 PixInsight 一条龙
├── src/
│   ├── CoreMLEngine.h    CoreML 引擎接口(纯 C++)
│   ├── CoreMLEngine.mm   引擎实现:compileModelAtURL / MLModel / MLMultiArray
│   ├── FKAstroProcess.h  进程与参数元类定义
│   ├── FKAstroProcess.cpp  分块推理管线(FKv3 / 旧 FKv2 方案)
│   ├── FKv3Pipeline.h    可离线测试的 FKv3 归一化、参数、分块与融合
│   ├── TensorLayout.h    按 strides 访问张量
│   ├── FKCurves.h        FKv2 标定曲线库(5 组 × 498 节点,BXT ML5 运行时采集)
│   ├── FKAstroInterface.h   自定义界面定义(五区分组)
│   ├── FKAstroInterface.cpp 界面实现(Options 区含处理方案下拉)
│   └── FKAstroModule.cpp 模块入口(MetaModule + InstallPixInsightModule)
├── build/                编译中间产物(按架构分子目录)
└── dist/FKAstro-pxm.dylib  ← 最终模块
```

## 构建

前置条件:
- `/Applications/PixInsight` 完整安装(自带 PCL 2.11 头文件和源码)
- Xcode(MacOSX SDK)

```bash
cd ~/workplace/fk-astro
make -j8            # 编译第三方库 + PCL + 模块(首次约 10–20 分钟)
```

产出 `dist/FKAstro-pxm.dylib`(arm64 + x86_64 通用二进制,ad-hoc 签名)。
增量重建只编模块源码时:`make -j8` 会自动跳过未变化的部分。

## 安装到 PixInsight

PixInsight **强制要求模块签名**(`.xsgn`,Ed25519),未签名模块无法安装。
自用走 Local Signing Identity(绑定你的 PixInsight 许可证)。

一键流程(先在 PixInsight `Edit > Local Signing Identity...` 创建过本地签名身份):

```bash
XSSK=~/workplace/fk-astro/dev.xssk ./install.sh
# 提示输入 .xssk 密码;或 XSSK_PASSWORD=... ./install.sh
```

脚本会:签名 → `sudo cp` 到 `/Applications/PixInsight/bin/` → 以 `-m` 参数
启动 PixInsight 加载模块。**PixInsight 不能在运行中**(先正常退出再跑脚本),
首次加载后正常退出一次,让模块安装状态落盘。

手动等价流程:

```bash
/Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight \
    --sign-module-file=$HOME/workplace/fk-astro/dist/FKAstro-pxm.dylib \
    --xssk-file=<你的密钥文件.xssk> --xssk-password="<密码>"
sudo cp dist/FKAstro-pxm.dylib dist/FKAstro-pxm.xsgn /Applications/PixInsight/bin/
```

发布给其他人用则需申请 CPD(Certified PixInsight Developer),
详见本机文档 `/Applications/PixInsight/doc/docs/CodeSigning/`。

## 使用

进程名 **FKAstroInference**(分类 Preprocessing / FKAstro),自定义界面分五区:
Model / Stellar Adjustments / Nonstellar Adjustments / Options /
Additional Options。

| 参数 | 默认 | 说明 |
|---|---|---|
| modelPath | — | 模型文件路径(.mlpackage 或 .mlmodel) |
| sharpenStars | 0.5 | 恒星区锐化强度 [0, 0.7](参数槽 0) |
| adjustStarHalos | 0.0 | 恒星光晕调整 [-0.5, 0.5](参数槽 1) |
| psfDiameter / autoPSF | 0.0 / true | 目标 PSF 直径 [0, 8] px;自动时模型自估(参数槽 2/3) |
| sharpenNonstellar | 0.5 | 非恒星区锐化强度 [0, 1](参数槽 4) |
| correctOnly | true | 中性校正模式:槽 0–4 清零(自动 PSF 仍生效) |
| lunarPlanetary | false | FKv3 中禁用恒星/自动 PSF，使用手动 PSF 和 x=1；Correct Only 覆盖此模式 |
| normScheme | FKv3 | 处理方案（Processing scheme）,见下 |
| reflectPad | true | 仅旧方案使用。FKv3/FKv2 固定零填充，FKv3 只归一化有效区域 |
| tileOverlap | 0.2 | FKv3 同时按此比例计算块数和偶数余弦带长度，范围 [0,0.5] |
| device | Auto | Auto / GPU / CPU |
| verbose | true | 过程控制台输出进度 |

### 处理方案(Options 区)

界面通过 **Processing scheme** 选择宿主处理方案；模型文件仍由 **Model path** 单独指定。
`FKv2/FKv3` 是本项目方案名，`ML5` 指官方模型版本。参数 ID 保持 `normScheme`，
枚举值为 `None=0/FKv2=2/FKv3=3`(值 1 为已移除的 FKv1,历史编号不复用;
旧 FKv1 实例执行时会报错,需重新选择方案)。

- **FKv3**（新实例默认，按恢复的官方 ML5 调用约定实现）— 按整图拟合 `arcsinh,wp=1`，第六列为同一状态的
  `log10(x/asinh(x))`。裁边 5%、长边 128 最近邻正样本、low 为 0.1% 分位、
  high=1、中位数决定 x。越界直接填模型域 0；所有块融合后再逆变换及 clamp。
  要求 1024×1024×3 输入、中心 512×512×3 输出及六个参数。
  旧进程实例仍使用保存的方案；需要此流程时将 Processing scheme 切换为 FKv3。
  公式、模式规则和验证范围见 [FKv3 说明](docs/FKv3.md)。

- **None** — 像素直通模型,参数向量只含 5 个用户槽位。
  BXT **ML4(5 参数模型)用这个**:官方 ML4 的归一化接近恒等,
  直通即可到全图 mean 差 ~8e-5 的精度(见下表)。
- **FKv2**（旧标定方案） — 标定曲线归一化。按图测 `bg = P35 分位数`,
  第 6 列同样是 `log10gain`。正/逆变换换成从 **BlurXTerminator ML5 运行时真实模型
  输入**采集的单调曲线(`src/FKCurves.h`):
  - 曲线按 `log(bg)` 最近邻从 5 组标定中选择
    (bg = 0.002 / **0.0102 深空典型** / 0.1025 / 0.19 / 0.35,
    每组 498 节点,log-x 均匀重采样加密尾部);
  - 插值为 **PCHIP(Fritsch-Carlson 单调三次)**,与 scipy
    `PchipInterpolator` 逐位一致;正逆两套节点斜率独立构建,
    逆变换在端点饱和钳位(模型输出超出标定域时取端点,不外推);
  - 归一化只作用于窗内有效(非零填充)区域。
  同样要求 `[1, 6]` parameters 输入,即只适用于 ML5。

### 分块网格

FKv3 路径设输出块 T=512，先计算偶数窗长 `L=2×floor(float32(T)×float32(overlap)/2)`，
再计算 `n=ceil(D/(T−L))` 和 `pitch=float32(D/n)`。首末块贴边，内部原点按官方
float32 运算后转 double 取整；小图保留负原点以居中。

余弦过渡带位于等间距接缝附近；带外多余覆盖置零，整图外边界不衰减。
输出在 double 模型域直接加权累加，整图只逆变换一次，不除以权重和。
FKv2 仍走旧网格和逐块逆变换，不能代表新 FKv3 逻辑。

## BXT 复现配方与精度

当前 FKv3 路径设置：

| 设置 | 值 |
|---|---|
| modelPath | 解密后的明文 `BlurXTerminator.5.mlpackage`(见 `~/py-astro`) |
| normScheme | **FKv3** |
| sharpenStars / adjustHalos / psf / autoPSF / sharpenNonstellar | 与官方 BXT 界面同值 |
| correctOnly | 按需(false = 锐化,true = 仅校正) |
| tileOverlap | 0.2(与官方默认一致) |

以下为企鹅 master 的历史 FKv2 实测表，**不适用于当前 FKv3 路径**，本轮未复测。

M31 单帧(6248×4176, 16bit, ML5, sharpen 0.56/0.37, halo 0.10, autoPSF)
对官方 BlurXTerminator 2.6.9 的全图像素差([0,1] 域):

| 配置 | P50 | P99 | P99.9 | 亮区(a>0.03)相对 | d>0.3 占比 |
|---|---|---|---|---|---|
| **ML5 + FKv2** | **0**（P50，不代表全图逐位相同） | 2.9e-4 | 5.7e-3 | 3.4% | 0.0000% |
| ML4 + None | 1.5e-5 | 7.0e-4 | — | 6.1% | — |

ML4 结论:官方 ML4 归一化近恒等(5 参数模型,无背景槽),
None 直通即达全图 mean 差 7.8e-5;若要 ML4 也到逐位级,
需另外采集 ML4 标定曲线(当前曲线库只覆盖 ML5)。

上表是在第 6 列仍为 `-log10(bg)`、重叠区为线性羽化时测得的。
当前 FKv3 路径使用配套的 arcsinh/gain、官方接缝和融合后逆变换；需要在 Mac 上重新对比。

## 模型要求(当前骨架的限制)

- FKv3 方案要求 1024×1024×3 输入、512×512×3 输出、6 参数。
- 通用方案支持固定形状的 2/3/4 维浮点张量（batch 须为 1），CHW/HWC 自动识别，
  按实际 strides 访问；float16/32/64 在引擎边界转换，内部模型缓冲为 float32。
- 支持输出等大、中心裁切和放大几何；通用模型应按其训练约定选择处理方案。
- 通道数:模型 C == 图像通道数,或 C == 1(单通道模型逐通道独立处理)。
- FKv3 小图在输出块中居中，并给输入上下文补模型域 0；单通道 CFA 按灰度平面处理，不进行去马赛克。
- 像素约定：浮点像素直接使用；整数图像先按最大采样值归一到 [0,1]。
  FKv3 写回时缩放到整数范围并按 nearest/even 取整，浮点图直接转换到目标浮点类型。

## 开发要点(PixInsight 模块开发的已验证实践)

- 模块命名 `<Name>-pxm.dylib`,install_name 为
  `@executable_path/FKAstro-pxm.dylib`
- 链接:`-lPCL-pxi -llz4-pxi -lzstd-pxi -lzlib-pxi -lRFC6234-pxi -llcms-pxi`
  + `-framework AppKit -framework Foundation -framework CoreML`
- FKv3 输出块 512²，输入 1024² 提供上下文；小图允许负原点，回写裁出有效交集。
- FKv3 状态和融合在 double 域计算，喂模型前转 float32；
  逆变换后写回整型图像时必须过 `HighestSampleValue` 量化,
  不能把 [0,1] 值直接标量转换成整型(会全黑)


## 离线验证

```bash
python tests/run_offline.py
```

编译生产 FKv3 数学实现及生产图像 reader/回写函数，运行 19 类宿主测试，包含不同三通道输出的逐块单色合并和原始 CFA 平面读取；不需要模型文件。
Windows 需要 Visual Studio C++ Build Tools，macOS/Linux 需要 C++17 编译器。
本轮未编译完整 macOS/PCL 模块，也未执行 CoreML 或官方成图对比，详见 [验证范围](docs/FKv3.md)。
