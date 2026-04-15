import argparse
import time
from funasr import AutoModel

def main():
    parser = argparse.ArgumentParser(description="FunASR 长音频转写脚本")
    parser.add_argument("audio_path", type=str, help="输入音频文件路径 (WAV格式，16kHz单声道)")
    parser.add_argument("--model", type=str, default="paraformer-zh", help="识别模型，默认 paraformer-zh")
    parser.add_argument("--batch_size_s", type=int, default=300, help="按秒数分批次，默认300秒")
    args = parser.parse_args()

    print(f"加载模型... (首次运行会自动下载)")
    model = AutoModel(
        model=args.model,
        vad_model="fsmn-vad",
        punc_model="ct-punc",
        device="cuda:0",
        disable_update=True,
        batch_size_s=args.batch_size_s,
    )

    print(f"开始识别: {args.audio_path}")
    start_time = time.time()
    res = model.generate(input=args.audio_path)
    elapsed = time.time() - start_time

    text = res[0]["text"]
    print("\n识别结果：")
    print(text)
    print(f"\n耗时: {elapsed:.2f} 秒")

    # 可选：将结果保存到文本文件
    output_txt = args.audio_path.rsplit(".", 1)[0] + ".txt"
    with open(output_txt, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"结果已保存至: {output_txt}")

if __name__ == "__main__":
    main()
