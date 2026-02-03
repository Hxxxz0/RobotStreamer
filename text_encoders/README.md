# Text Encoders

可选：将文本编码器放在这里，避免每次从 Hugging Face 下载。

## T5 编码器（默认）

```bash
# 复制已有模型
cp -r /limx_embap/tos/user/Jensen/project/MotionStreamer/flan-t5-small \
      text_encoders/
```

## BGE 编码器

```bash
cd text_encoders
git clone https://huggingface.co/BAAI/bge-small-en-v1.5
```

## 自动下载

如果为空，首次运行会自动从 Hugging Face 下载。

---

**目录结构：**
```
text_encoders/
├── README.md
├── flan-t5-small/        ← T5（可选）
└── bge-small-en-v1.5/    ← BGE（可选）
```
