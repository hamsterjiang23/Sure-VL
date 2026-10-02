# 第三方源码快照

这里保存三个上游仓库在固定提交上的**完整已跟踪文件**。它们直接位于 Sure-VL 仓库中，不是 Git 子模块；克隆 Sure-VL 后即可阅读和编辑。快照通过 `git archive` 取得，不包含上游 `.git`、本地环境、模型权重或数据缓存。`SOURCES.json` 记录每个文件的哈希、权限、来源和提交。

| 目录 | 上游 | 固定提交 | 仓库级许可证 |
| --- | --- | --- | --- |
| [`OPSD/`](OPSD/) | [siyan-zhao/OPSD](https://github.com/siyan-zhao/OPSD) | `ae7d2519e94920c4eb6206c0c26de46d9c50abae` | 未提供（unknown） |
| [`Vision-OPD/`](Vision-OPD/) | [VisionOPD/Vision-OPD](https://github.com/VisionOPD/Vision-OPD) | `06860e69b5ed9dc24e96ca5c855f3a4ef25976aa` | [Apache-2.0](Vision-OPD/LICENSE) |
| [`VL-Calibration/`](VL-Calibration/) | [Mr-Loevan/VL-Calibration](https://github.com/Mr-Loevan/VL-Calibration) | `d38e15869d3a95d5d0c8fa1627d9745e490685d6` | [Apache-2.0](VL-Calibration/LICENSE) |

OPSD 在该提交中没有仓库级 `LICENSE` 文件，也没有在 README 中声明整体许可。其 `opsd_trainer.py` 保留了原有的 HuggingFace Apache-2.0 文件头；该文件头不代表 OPSD 其余文件均采用 Apache-2.0。这里不对 OPSD 的整体授权作推断。

`python third_party/verify_sources.py` 可核对三个目录是否仍与固定上游提交的文件内容一致。修改这里的源码时，直接编辑对应文件，并用 Sure-VL 的 Git 提交记录本地改动；校验脚本会将这些改动列为 `modified`，便于区分上游基线和本地修改。

更新上游版本时，先在独立临时目录检出新提交，用 `git archive <commit> | tar -xf - -C <空临时目录>` 制作新快照；与当前目录比较并审查本地改动后，再替换对应目录，更新此表、`configs/reference_sources.json` 和 `SOURCES.json` 中的提交及哈希。不要直接在现有目录解包新版本：上游删除的文件会被遗留。
