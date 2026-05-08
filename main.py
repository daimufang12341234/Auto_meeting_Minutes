from funasr import AutoModel
from pathlib import Path


def main():
    # 加载一体化模型：ASR + VAD + 标点 + 说话人分离
    model = AutoModel(
        model="paraformer-zh",
        model_revision="v2.0.4",
        vad_model="fsmn-vad",
        vad_model_revision="v2.0.4",
        punc_model="ct-punc-c",
        punc_model_revision="v2.0.4",
        spk_model="cam++",
        spk_model_revision="v2.0.2",
        device="cuda:0",
        disable_update=True,
    )

    # 使用测试音频
    wav_path = "F:/pyproject/auto_conference_minutes/example/录音.wav"

    # 生成结果（包含说话人分离）
    res = model.generate(
        input=wav_path,
        batch_size_s=300,
    )

    # 打印完整结果供调试
    print("=" * 60)
    print("原始输出字段：")
    print("=" * 60)
    for key, value in res[0].items():
        if isinstance(value, str) and len(value) > 100:
            print(f"  {key}: {value[:100]}...")
        elif isinstance(value, list) and len(value) > 3:
            print(f"  {key}: {value[:3]}... (共{len(value)}项)")
        else:
            print(f"  {key}: {value}")

    # 解析并保存结果
    result = res[0]
    sentence_info = result.get("sentence_info", [])

    if sentence_info:
        # 生成格式化输出
        output_lines = format_output(sentence_info)

        # 打印结果
        print("\n" + "=" * 60)
        print("会议转录结果（按说话人区分）：")
        print("=" * 60)
        for line in output_lines[:10]:  # 只打印前10行示例
            print(line)
        if len(output_lines) > 10:
            print(f"... 共 {len(output_lines)} 行")

        # 保存到文件
        output_path = Path(wav_path).with_suffix(".transcript.txt")
        save_output(output_lines, output_path)
        print(f"\n结果已保存到: {output_path}")
    else:
        print(f"\n转录文本：{result.get('text', '')}")


def format_output(sentence_info: list) -> list:
    """将 sentence_info 转换为目标格式"""
    lines = []
    for i, sentence in enumerate(sentence_info, start=1):
        # 时间从毫秒转换为秒
        start_sec = sentence.get("start", 0) / 1000
        end_sec = sentence.get("end", 0) / 1000

        # 说话人ID (FunASR用0开始，参考格式用1开始)
        spk_id = sentence.get("spk", 0) + 1

        # 文本内容
        text = sentence.get("text", "")

        # 格式: 序号	[开始秒,结束秒]	说话人ID	性别,语言	文本
        # FunASR不提供性别信息，默认填"未知,普通话"
        line = f"{i}\t[{start_sec:.3f},{end_sec:.3f}]\t{spk_id}\t未知,普通话\t{text}"
        lines.append(line)

    return lines


def save_output(lines: list, output_path: Path):
    """保存结果到文件"""
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")


def merge_speaker_segments(segments: list):
    """合并同一说话人的连续话语"""
    merged = []
    current_speaker = None
    current_text = []

    for speaker, text in segments:
        if speaker != current_speaker:
            if current_speaker and current_text:
                merged.append((current_speaker, "".join(current_text)))
            current_speaker = speaker
            current_text = [text]
        else:
            current_text.append(text)

    if current_speaker and current_text:
        merged.append((current_speaker, "".join(current_text)))

    return merged


if __name__ == "__main__":
    main()