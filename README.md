# fk-astro

在 PixInsight 里运行你自己训练的 CoreML 模型(`.mlpackage` / `.mlmodel`)的
最小可用模块骨架。分块推理 + 重叠边距,加载你自己训练的 CoreML 模型
(明文 .mlpackage/.mlmodel 直读,无需任何中间层)。

## 目录结构

```
fk-astro/
├── Makefile              构建脚本(双架构通用二进制)
├── src/
│   ├── CoreMLEngine.h    CoreML 引擎接口(纯 C++)
│   ├── CoreMLEngine.mm   引擎实现:compileModelAtURL / MLModel / MLMultiArray
│   ├── FKAstroProcess.h  进程与参数元类定义
│   ├── FKAstroProcess.cpp  分块推理管线(FillTile/Predict/StoreTile/FKv1 归一化)
│   ├── FKAstroInterface.h   自定义界面定义(五区分组)
│   ├── FKAstroInterface.cpp 界面实现(Options 区含归一化下拉)
│   └── FKAstroModule.cpp 模块入口(MetaModule + InstallPixInsightModule)
├── PIPELINE.md           推理流程通用规范(归一化/6 参数契约)
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
自用走 Local Signing Identity(绑定你的 PixInsight 许可证):

1. 打开 PixInsight,菜单 `Edit > Local Signing Identity...`,
   按提示创建本地签名身份(会生成密钥文件,记住密码)。
2. 用命令行给模块签名:

   ```bash
   /Applications/PixInsight/PixInsight.app/Contents/MacOS/PixInsight \
       --sign-module-file=$HOME/workplace/fk-astro/dist/FKAstro-pxm.dylib \
       --xssk-file=<你的密钥文件.xssk> \
       --xssk-password="<密码>"
   ```

   生成 `dist/FKAstro-pxm.xsgn`。
3. 复制到模块目录并重启 PixInsight:

   ```bash
   sudo cp dist/FKAstro-pxm.dylib dist/FKAstro-pxm.xsgn /Applications/PixInsight/bin/
   ```

   (也可用 PixInsight 的模块安装界面安装。)

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
| lunarPlanetary | false | Lunar/Planetary 模式开关。只随进程保存,不写入 parameters 的 6 列 |
| normScheme | None | 归一化方案,见下 |
| reflectPad | true | 分块窗口越界时镜像填充(否则复制边缘像素)。FKv2 越界填 0 |
| tileOverlap | 0.2 | 相邻块重叠比例 [0, 0.5]。中心裁切的块原点仍是等间距网格,接缝用升余弦,窗长取中间块的实际间距 |
| device | Auto | Auto / GPU / CPU |
| verbose | true | 过程控制台输出进度 |

### 归一化(Options 区)

- **None** — 像素直通模型,参数向量只含 5 个用户槽位。
- **FKv1** — 逐块像素曲线仍是
  `log → MTF(m=0.100) → affine(a=1.4685, b=-0.7095)`,工作点
  `wp = 0.505 × bg`,`bg` 是正像素的 P35。
- **FKv2** — 用同一份 P35 在标定曲线里选一条,曲线在 `src/FKCurves.h`。
  越界像素填 0。
- 两种方案的第 6 列都是 `log10gain`,不再是 `-log10(P35)`。要求模型声明
  `[1, 6]` parameters 输入。

## 模型要求(当前骨架的限制)

- 输入输出都是 **固定形状 3-D float32 多数组**(CHW 或 HWC 自动识别)。
- **输出形状 == 输入形状**(去噪/锐化类模型)。超分类模型需要自行扩展
  `StoreTile` 的映射。
- 通道数:模型 C == 图像通道数,或 C == 1(单通道模型逐通道独立处理)。
- 图像小于模型输入时自动零填充单块处理。
- 像素约定:PixInsight 归一化浮点 [0,1] 直接喂给模型并直接写回
  (训练时请用相同的 [0,1] 归一化;其他值域需自行在源码中加变换)。

## 开发要点(PixInsight 模块开发的已验证实践)

- 模块命名 `<Name>-pxm.dylib`,install_name 为
  `@executable_path/FKAstro-pxm.dylib`
- 链接:`-lPCL-pxi -llz4-pxi -lzstd-pxi -lzlib-pxi -lRFC6234-pxi -llcms-pxi`
  + `-framework AppKit -framework Foundation -framework CoreML`
- 分块原点: `n = ceil(size / (tile*0.8))`,`sx = size/n`,中间块
  `round(sx*k - (tile-sx)/2)`,第一块 0,最后一块贴边。
  混合权重是升余弦,窗长等于中间两块的实际重叠。输出只写每块中心区域。
