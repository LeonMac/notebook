ffmpeg命令，提取mov文件内的音频，从第3m5s开始，到46m5s结束。
我们用户需要ffmpeg命令来提取mov文件中的音频，从第3分5秒开始到46分5秒结束。需要指定时间范围。命令格式：ffmpeg -i input.mov -ss 00:03:05 -to 00:46:05 -c copy output.aac 或者重新编码。通常提取音频可以用-c copy如果音频编码是兼容的，但可能需要指定音频编码。更通用的是使用-c copy或-acodec。注意时间格式：3m5s => 00:03:05，46m5s => 00:46:05。最好使用-ss和-to选项。另外如果使用-c copy，需要注意关键帧问题，但-ss放在-i之前可以快速定位但可能不精确，放在-i之后可以精确但速度慢。通常对于提取音频，精确很重要，可以接受重新编码。建议：ffmpeg -i input.mov -ss 00:03:05 -to 00:46:05 -q:a 0 output.mp3 或者 -c copy output.m4a 等。但用户没指定输出格式，我们可以给出常见选项。提供清晰命令。

你可以使用以下 FFmpeg 命令从 MOV 文件中提取第 3 分 5 秒到第 46 分 5 秒的音频：

bash
ffmpeg -i input.mov -ss 00:03:05 -to 00:46:05 -q:a 0 output.mp3
参数说明：

-i input.mov ：输入文件。

-ss 00:03:05 ：开始时间（3分5秒）。

-to 00:46:05 ：结束时间（46分5秒）。

-q:a 0 ：音频质量（0 最好，适用于 MP3 编码）。

output.mp3 ：输出文件名，可改为 .aac、.m4a 等。

其他常用输出格式示例：

保留原始音频编码（最快，无质量损失）：

bash
ffmpeg -i input.mov -ss 00:03:05 -to 00:46:05 -c copy output.m4a
注意：如果原始音频不是 MOV 支持的封装格式，可能需调整输出扩展名。

输出高质量 AAC：

bash
ffmpeg -i input.mov -ss 00:03:05 -to 00:46:05 -c aac -b:a 192k output.aac
时间格式说明：
00:03:05 表示 0 小时 3 分 5 秒，你也可以使用纯秒数，例如 -ss 185（3×60+5=185 秒）。

有没有本地化模型，可以识别mp3内容（中文）并将其转化成文字？
用户想了解是否有本地化模型可以识别MP3内容（中文）并将其转换为文字。这涉及到中文语音识别（ASR）的本地化解决方案。为了全面覆盖，我需要同时进行多项搜索，涵盖不同的关键词和可能的平台。

搜索到 25 个网页
搜索结果显示了一些相关的模型和工具。为了全面了解这些信息，我们需要同时打开所有看起来相关的链接。

浏览 10 个页面
Whisper v0.2 下载安装教程：免费本地语音转文字工具下载安装使用全流程
FunASR中文语音识别音频转文本声音转文本系统 二次webui开发构建by科哥一键部署
Vosk离线中文语音识别：从部署到优化的全流程指南
离线语音转文字新选择：FunASR本地化部署指南
基于Vosk的中文离线语音识别：从部署到实战全解析
查看全部
这些结果列出了多个本地化的中文语音识别模型。我的回答将介绍几款主流模型，包括Whisper、FunASR、Vosk、GLM-ASR、Qwen3-ASR等，并提供一个对比表格，然后补充通用步骤和注意事项。

有的，有不少优秀的本地化模型可以实现这个需求。根据你的场景（将 MP3 转写成文字），这几种主流方案各有特点：

模型 / 工具	特点	中文识别准确率	资源要求	上手难度
Whisper (OpenAI)	通用性强，支持多语言，社区生态完善，有图形界面版本可用。	较高 (具体依赖模型大小)	高 (模型大，对显存有要求)	低 (有图形界面版，新手友好)
FunASR (阿里达摩院)	中文识别专精，准确率极高，工业级方案，有现成WebUI。	非常高 (可达95%)	中等 (8GB内存可跑，推荐NVIDIA显卡)	低 (Docker或整合包，相对简单)
Vosk	极轻量、跨平台，适合低功耗设备（如树莓派），无需GPU。	较高 (约90%)	低 (CPU可跑，最小模型仅50MB)	中等 (需Python编程)
GLM-ASR (智谱AI)	复杂场景鲁棒性强，专为低音量、带口音等复杂音频优化，性能领先。	非常高 (在中文基准测试中表现优异)	较高 (1.5B参数)	中等 (需Python编程)
Qwen3-ASR (通义千问)	统一模型同时支持流式与离线识别，支持52种语言及22种中文方言。	非常高 (SOTA水平)	较高	中等 (需Python编程)
💡 如何选择？
如果你的核心需求是高精度的中文识别，且电脑配置允许，FunASR 是首选。如果想快速体验，Whisper 因为有图形界面版，对新手非常友好。Vosk 则更适合在低配置环境或嵌入式设备中运行。

🛠️ 通用操作步骤与注意事项
准备音频：确保你的 MP3 文件采样率为 16kHz，格式为 单声道 的 WAV/PCM。如果不是，可以用 ffmpeg 转换。

模型选择：根据你的电脑配置（特别是 GPU 显存）选择合适的模型大小。模型越大，通常越准，但也越慢。

安装部署：

新手可以从带有图形界面的版本开始，如 WhisperDesktop 或 FunASR 的整合包。

熟悉代码的开发者，可通过 pip 安装对应的 Python 库，几行代码即可实现核心功能。

开始识别：将准备好的音频文件导入软件，等待模型处理即可得到文字稿。

💡 关于音频格式：很多模型原生支持 WAV 格式。如果你的音频是 MP3，需要先转换。这里有个转换命令，可以作为参考：

bash
# 将 MP3 转换为 16kHz 单声道的 WAV 文件
ffmpeg -i input.mp3 -acodec pcm_s16le -ac 1 -ar 16000 output.wav
如果对某个模型的部署细节有疑问，或者需要具体的代码示例，可以随时再问我～

我有本地显卡 NVIDIA GeForce GTX 1050Ti，请介绍本地化python代码实现中文识别的方案。
用户想知道如何在本地使用NVIDIA GeForce GTX 1050Ti显卡，通过Python代码实现中文语音识别。我需要提供具体的方案，包括模型选择、安装步骤、代码示例等。考虑到用户是中文用户，我应该推荐适合中文识别的模型，比如FunASR、Whisper、Vosk等。我需要搜索关于在GTX 1050Ti上运行这些模型的相关信息。

搜索到 42 个网页
搜索结果显示了一些相关页面。为了全面了解不同模型（如FunASR、Whisper、Vosk）在GTX 1050Ti上的使用情况，我们需要同时打开这些结果。

浏览 10 个页面
开源可部署！SenseVoice-Small ONNX语音识别工具GPU/CPU双适配实操手册
应用Python+TensorFlow完成人工智能中文语音辨认 - 智客公社
基于funasr+pyaudio实现电脑本地麦克风实时语音识别项目语音转文本python实现
10分钟上手FunASR语音大数据分析：从录音到用户行为洞察全流程
Python Whisper实时语音识别：从理论到实践的全链路解析
查看全部
这些搜索结果提供了多种方案。我将组织回答，首先提供针对GTX 1050Ti的模型选型建议，然后提供Python代码实现和示例，最后进行总结。

基于你的显卡（GTX 1050Ti， 4GB显存），最适合本地的方案是使用阿里达摩院的 FunASR 或 Vosk，它们对中文的识别效果都很好。

为方便你选择，这里有一份简要的对比：

特性	FunASR (推荐)	Vosk	Whisper
中文识别	极高 (98%+)，工业级	较高 (约90%)	高 (默认)
硬件资源	适中，4GB显存可用	低，纯CPU亦可运行	高，4GB显存仅能运行极小或部分量化模型
主要特点	专为中文优化，集成VAD、标点恢复	轻量，完全离线，适合低配设备	多语言通用，但资源消耗大
方案一：FunASR（中文识别首选）
这是最适合你的方案，集成了语音端点检测（VAD）和标点恢复，能自动处理长音频，识别效果很好。

1. 环境配置

打开终端或命令行，依次执行以下命令：

bash
# 创建独立环境 (推荐)
conda create -n funasr python=3.8
conda activate funasr

# 安装 PyTorch (CUDA 11.8 版本，适配你的显卡)
pip3 install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装 FunASR
pip3 install -U funasr modelscope
2. 基础使用

创建一个 Python 文件（如 transcribe_funasr.py），复制以下代码：

python
from funasr import AutoModel

# 1. 加载模型 (首次运行会自动下载)
model = AutoModel(
    model="paraformer-zh",      # 中文识别模型
    vad_model="fsmn-vad",       # 语音端点检测，用于长音频分割
    punc_model="ct-punc",       # 标点恢复模型，添加标点符号
)

# 2. 识别音频
res = model.generate(input="你的音频.wav")
print(res[0]["text"])
3. 核心参数说明

model: 指定识别模型。FunASR 还提供 paraformer-zh-streaming 实时模型。

vad_model: 语音端点检测，能自动分割长音频，提升处理效率。

punc_model: 标点恢复，让识别结果更规范。

完整的高级用法和更多模型选项可参考其官方文档。

方案二：Vosk（轻量高效之选）
如果你的主要目标是轻量、快速，不希望依赖 CUDA，Vosk 是绝佳的备选方案。

1. 环境配置

bash
pip install vosk
2. 模型下载

从 Vosk 官网下载中文模型：

轻量版 (40MB): vosk-model-small-cn-0.22

高精度版 (1.1GB): vosk-model-cn-0.22

3. 基础使用

python
import wave
import json
from vosk import Model, KaldiRecognizer

# 1. 加载模型
model = Model("路径/到/你的/vosk-model-cn-0.22")
# 2. 打开音频文件（需为单声道，16kHz的WAV格式）
wf = wave.open("你的音频.wav", "rb")
# 3. 创建识别器
rec = KaldiRecognizer(model, wf.getframerate())

# 4. 逐帧识别并获取结果
while True:
    data = wf.readframes(4000)
    if len(data) == 0:
        break
    if rec.AcceptWaveform(data):
        result = json.loads(rec.Result())
        print(result["text"])
💎 其他模型
SenseVoice: 提供 ONNX 量化版，对硬件更友好，支持中英文混合识别。

Whisper: 考虑到显存限制，可以尝试 tiny 或 base 版本，但体验可能不及 FunASR。

💡 音频预处理建议
为提高识别效果，建议先将音频转为 16kHz 单声道 WAV 格式：

bash
ffmpeg -i 你的音频.mp3 -acodec pcm_s16le -ac 1 -ar 16000 你的音频.wav
📌 注意：请确保你的显卡驱动和 CUDA 环境配置正确，这样才能发挥 GPU 的加速效果。如果你在安装 FunASR 时遇到问题，可以告诉我具体的报错信息，我会帮你排查。

我目前已经安装了 NVIDIA-SMI 580.126.09             Driver Version: 580.126.09     CUDA Version: 13.0。
这可以跑FunASR吗？
用户想知道，在安装了NVIDIA-SMI 580.126.09、驱动版本580.126.09和CUDA版本13.0的情况下，能否运行FunASR。我需要提供相关信息。为了全面回答这个问题，我需要同时进行多项搜索，以获取FunASR的CUDA兼容性要求、PyTorch与CUDA 13.0的兼容性、以及GTX 1050 Ti的算力等信息。

搜索到 31 个网页
搜索结果显示，FunASR的CUDA兼容性要求可能没有明确提及。关于PyTorch与CUDA 13.0的兼容性，结果0显示了一个自定义的PyTorch 2.10.0a0版本，结果1提到CUDA 13.2和12.6，结果3提到PyTorch 2.11.0默认安装CUDA 13.0，但同时也提到不再支持Maxwell和Pascal GPU。关于GTX 1050 Ti的计算能力，结果0和结果1都显示其计算能力为6.1（Pascal架构）。关于NVIDIA驱动580和CUDA 13.0的支持，结果0提到CUDA 13.0支持Turing到Blackwell架构，但可能不包括Pascal。结果1提到CUDA 13.0搭配580驱动。结果3提到R580驱动支持CUDA 13.0。为了全面了解GTX 1050 Ti在CUDA 13.0下的兼容性，以及PyTorch对CUDA 13.0的支持情况，我们需要同时打开结果0、结果1、结果3、结果2、结果5进行详细查看。

浏览 6 个页面
FunASR实战指南：语音识别模型从训练到微调的全流程解析
Introducing CUDA 13.2 and Deprecating CUDA 12.8 (Release 2.12)
torch 2.11.0 on Python PyPI
NVIDIA CUDA Toolkit 13.0 Is Out
NVIDIA CUDA 13.0 U2 Brings DGX Spark Performance Improvements
查看全部
这些信息表明，FunASR 可能支持 CUDA 11.x，而 PyTorch 2.11 默认使用 CUDA 13.0，但不再支持 Pascal 架构（GTX 1050 Ti 属于 Pascal 架构）。CUDA 13.0 仅支持 Turing 及更新架构。用户需要安装 CUDA 12.6 版本的 PyTorch。因此，答案是否定的。我的回答将首先给出一个简短结论，然后通过一个对比表格直观展示问题所在，接着用通俗易懂的方式解释核心原因，最后提供一个从卸载到安装的完整操作方案，并在末尾留出进一步协助的开口。

结论是：可以运行，但不能直接使用最新的默认版本，需要手动安装一个特定的 PyTorch 版本（CUDA 12.6）。

你目前的 CUDA 13.0 驱动版本，以及你的显卡 GTX 1050Ti（Pascal 架构），在默认配置下确实存在兼容性问题。

你可以通过下表快速理解核心原因：

事项	说明	结论
你的环境	GTX 1050Ti (Pascal架构) + CUDA 13.0 驱动	兼容性存在问题
PyTorch 默认版本	PyTorch 2.11 及以上，默认安装的版本不再支持 Pascal 架构	❌ 无法直接使用
解决方案	手动安装 PyTorch CUDA 12.6 的版本，该版本保留了对 Maxwell、Pascal 等旧架构的支持	✅ 完美兼容
📝 为什么会出现这种情况？
简单来说，这是一个新旧硬件和软件迭代的问题：

PyTorch 升级了：从 2.11 版本开始，它在 PyPI 上默认提供的预编译包是基于 CUDA 13.0 的。

CUDA 移除了旧支持：CUDA 13.0 为了提高性能，移除了对 Maxwell（5.x）和 Pascal（6.x） 等较旧显卡架构的支持。你的 GTX 1050Ti 的算力是 6.1，属于 Pascal 架构，正好在这一批“被移除支持”的列表中。

🛠️ 解决方案
好消息是，解决这个问题只需要一步：手动指定安装兼容的 PyTorch 版本。

请打开你的终端（或 Anaconda Prompt），并遵循以下步骤：

激活你的 FunASR 环境：

bash
conda activate funasr
卸载现有的 PyTorch（如果已安装）：

bash
pip uninstall torch torchaudio torchvision
重新安装兼容的 PyTorch 版本 (CUDA 12.6)：

bash
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu126
这个命令会从 PyTorch 官网下载支持 CUDA 12.6 的版本，它完美兼容你的 GTX 1050Ti 显卡。

验证安装：
安装完成后，在 Python 环境中运行以下代码，确保一切正常：

python
import torch
print(torch.__version__)
print(torch.cuda.is_available())  # 这行应该返回 True
print(torch.cuda.get_device_name(0))  # 这行应该能正确识别你的显卡
如果 torch.cuda.is_available() 返回 True，说明你的环境已经配置成功，可以开始使用 FunASR 了。

完成以上步骤后，你的 FunASR 应该就能顺利运行了。如果在验证环节遇到任何报错，随时告诉我，我会协助你进一步排查。

这个安装会和我原先的 CUDA 版本 （nvidia-smi 所得）有冲突吗？
用户想知道安装PyTorch CUDA 12.6版本是否与通过nvidia-smi显示的CUDA 13.0驱动冲突。我需要解释清楚驱动版本与运行时CUDA版本之间的关系。为了全面获取信息，我需要同时进行多项搜索，涵盖中文和英文资料，以覆盖不同角度的解释。

搜索到 29 个网页
搜索结果显示了一些相关页面。为了全面了解情况，我们需要同时打开这些页面。

浏览 8 个页面
Ubuntu PyTorch与CUDA版本关系
Linux版PyTorch与CUDA版本如何匹配
What is the interplay between the "nvidia driver" and "cuda"?
Linux系统如何选择PyTorch版本
Python中PyTorch找不到GPU该怎么办_NVIDIA驱动检测与torch.cuda.is_available排查
查看全部
这些资料清楚地表明，nvidia-smi 显示的 CUDA 版本是驱动程序支持的最高版本，而 PyTorch 使用的是自己的 CUDA 运行时。用户安装的 PyTorch CUDA 12.6 版本是兼容的，因为驱动程序支持更高的版本。我的回答将首先给出一个清晰的结论，然后解释 nvidia-smi 显示的 CUDA 版本与 PyTorch 使用的 CUDA 版本之间的区别，接着解释 GTX 1050Ti 的兼容性，再提供验证方法，最后给出总结。

不会冲突，请放心。你的安装计划和当前环境是完美兼容的。

这个结论可能听起来有点反直觉，因为你之前刚得知“PyTorch 默认版本不再支持 GTX 1050Ti”，现在又说兼容。这是因为两者针对的是不同层面的“CUDA”，它们之间没有冲突。

你可以通过下表来理解它们的区别：

对比项	nvidia-smi 中的 CUDA 13.0	PyTorch 将安装的 CUDA 12.6
本质是什么	驱动程序的最高版本上限。	PyTorch 这个软件自带的 CUDA 工具集。
主要作用	告诉系统，当前这个显卡驱动最多能支持哪一代的 CUDA 软件。	PyTorch 在运行时实际调用的具体 CUDA 函数库。
相互关系	是一个“上限”或“保险丝”的角色。	只要它要求的版本 ≤ nvidia-smi 显示的版本，就能正常工作。
你的驱动版本（580.126.09）支持 CUDA 13.0 这个“上限”，而你准备安装的 PyTorch 只需要调用 CUDA 12.6 这个“具体指令”。这就像你的硬件系统明确表态“我能支持到 13.0”，而你要运行的软件只要求“我需要 12.6”。只要软件要求的版本不超过硬件支持的上限（13.0 >= 12.6），就能完美运行，这是向下兼容的标准做法。

💡 关于你的显卡 GTX 1050Ti 与 CUDA 12.6
你之前担心的“不支持”，其实是 PyTorch 新版预编译包为了性能优化，放弃了对较旧显卡架构（如你的 1050Ti 所用的 Pascal 架构）的兼容。但这并不代表硬件或驱动层面不支持。因此，我们选择手动安装一个对旧架构显卡支持更友好的 CUDA 12.6 版 PyTorch，就完美地绕开了这个问题。

✅ 验证方法
完成安装后，运行以下命令来验证一切正常：

python
import torch
print(torch.__version__)                # 应输出 PyTorch 版本号
print(torch.version.cuda)               # 应输出 '12.6'
print(torch.cuda.is_available())        # 应输出 True
print(torch.cuda.get_device_name(0))    # 应输出你的显卡型号
总结一下：nvidia-smi 中的版本是驱动支持的上限，它高于或等于 PyTorch 需要的版本就表示环境兼容。你现在可以放心按计划安装了。

安装完成，请建议下一步。
用户已经完成了PyTorch CUDA 12.6的安装，并验证了torch.cuda.is_available()返回True。现在需要提供下一步的指导，即安装FunASR并运行一个简单的测试脚本。为了全面获取相关信息，我们需要同时进行多项搜索，涵盖FunASR的安装、中文识别模型的使用、以及可能的错误处理。

搜索到 26 个网页
搜索结果显示了一些相关页面。为了全面了解FunASR的安装和使用方法，我们需要同时打开这些页面。

浏览 10 个页面
从零搭建语音识别系统：Fun-ASR + GPU算力平台
手把手教你部署Fun-ASR，本地ASR系统轻松搞定
从本地部署到结果导出｜FunASR语音识别全流程实操记录
Getting Started | modelscope/FunASR | DeepWiki
小白必看：Fun-ASR语音识别系统从0到1实操
查看全部
这些搜索结果提供了丰富的安装和使用信息。我将组织回答，涵盖安装 FunASR、验证 GPU、编写测试脚本、处理长音频、使用 WebUI 以及后续步骤。

很好，既然 PyTorch 已经装好了，下一步就是安装 FunASR 并让模型在你的机器上跑起来。

🚀 第一步：安装 FunASR
直接在终端中运行下面的命令，安装 FunASR 本体以及模型加载工具 modelscope：

bash
pip install -U funasr modelscope
💡 小贴士：考虑到你的 4GB 显存和长音频任务，建议在接下来的代码中加入 batch_size_s=300 参数来控制批处理大小，可以更稳定、高效地运行。

🧪 第二步：运行测试脚本
为了验证安装和环境是否完全就绪，你可以创建一个 .py 文件（比如 test_funasr.py），并写入下面的代码：

python
import torch
from funasr import AutoModel

# 1. 准备一个16kHz单声道的WAV音频文件，替换成你的文件路径
audio_file = "你的音频.wav"

# 2. 加载模型（首次运行会自动下载，请保持网络通畅）
model = AutoModel(
    model="paraformer-zh",       # 核心识别模型
    vad_model="fsmn-vad",        # 语音活动检测模型
    punc_model="ct-punc",        # 标点恢复模型
    device="cuda:0",             # 强制使用GPU加速
    disable_update=True,         # 禁止自动更新，保持环境稳定
    batch_size_s=300,            # 按300秒音频打包，优化长音频处理
)

# 3. 执行识别
res = model.generate(input=audio_file)

# 4. 打印结果
print("识别结果：", res[0]["text"])
运行这个脚本，看到打印出的文字，就说明安装成功了。你也可以直接使用命令行快速测试：

bash
funasr ++model=paraformer-zh ++input=你的音频.wav
如果运行时遇到任何报错，随时把错误信息发给我，我们一起排查。

💡 接下来的可选探索
处理长音频：你的 46 分钟音频可能会超过默认的 30 秒限制。解决方法很简单，只需在模型加载时设置 batch_size_s=300，FunASR 的 VAD（语音活动检测）就会自动处理长音频的切分和拼接。

使用 WebUI：如果觉得写代码太麻烦，可以试试启动 FunASR 的 Web 图形界面。它提供了更直观的操作和批量处理功能。部署成功后，在浏览器打开 http://localhost:7860 即可使用。

尝试其他模型：paraformer-zh 是通用模型。如果你需要实时识别（流式），可以换成 paraformer-zh-streaming；如果在复杂场景下效果不佳，也可以试试 SenseVoiceSmall。

测试20秒音频通过。现在准备正式转换这46分钟音频，请告知我详细步骤。请将音频文件路径作为参数传入脚本
我们确认用户已经测试通过20秒音频，现在需要处理46分钟音频。用户要求将音频文件路径作为参数传入脚本。需要提供详细步骤，包括如何运行脚本，参数传递方式，以及注意事项（如显存管理、VAD设置等）。需考虑用户GTX 1050Ti 4GB显存，batch_size_s=300已经建议过。提供完整脚本示例，使用argparse或sys.argv。同时提醒用户音频格式最好是16kHz单声道WAV，如果不是需要转换。给出完整命令。

很好，既然测试通过，接下来就可以处理46分钟的长音频了。我会提供一个完整的 Python 脚本，支持通过命令行参数传入音频文件路径，并针对你的 4GB 显存做了优化。

📝 完整脚本（保存为 transcribe.py）
python
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
🛠️ 使用步骤
1. 确保音频格式正确
FunASR 对采样率 16kHz、单声道、WAV 格式的音频效果最好。如果你的音频是 MP3 或其他格式，请先用 ffmpeg 转换：

bash
ffmpeg -i input.mp3 -acodec pcm_s16le -ac 1 -ar 16000 output.wav
注意：将 input.mp3 替换为你的原始文件，output.wav 替换为转换后的文件路径。

2. 运行脚本
打开终端（或 Anaconda Prompt），激活你的 FunASR 环境：

bash
conda activate funasr
然后执行脚本，将音频路径作为参数传入：

bash
python transcribe.py /path/to/your/audio.wav
示例：如果你的音频放在 D:\videos\speech.wav，则运行：

bash
python transcribe.py D:\videos\speech.wav
3. 查看结果
控制台会实时打印识别出的文字。

脚本会自动在与音频相同目录下生成一个同名的 .txt 文件，例如 speech.txt。

⚙️ 参数说明
参数	说明	默认值
audio_path	必需，音频文件路径	无
--model	模型名称，可选 paraformer-zh、paraformer-zh-streaming 等	paraformer-zh
--batch_size_s	按多少秒音频打包处理，长音频建议 300~600	300
如果你的 46 分钟音频在识别过程中出现显存不足（CUDA out of memory），可以尝试降低 --batch_size_s，比如：

bash
python transcribe.py /path/to/audio.wav --batch_size_s 200
💡 高级提示
实时进度：FunASR 默认不会打印进度条。如果想看到处理进度，可以在加载模型时添加 disable_log=False，但会增加输出信息。

保存为SRT字幕：如果需要生成带时间戳的字幕文件，可以改用 model.generate(..., output="srt")，具体参考 FunASR 文档。

断点续转：对于超长音频（例如几小时），建议先使用 ffmpeg 切分成 10-20 分钟的片段分别转写，再合并结果。

❓ 常见问题
Q：运行时提示 torch.cuda.OutOfMemoryError
A：降低 --batch_size_s 的值，比如 150 或 100。也可以尝试加载模型时加上 disable_log=True 减少内存开销。

Q：识别结果没有标点符号
A：检查模型加载时是否包含了 punc_model="ct-punc"。如果仍无标点，可能是音频中的语音停顿不明显，可以尝试使用 ct-punc-c 模型（更准确但更慢）。

Q：46分钟大概需要多长时间？
A：在 GTX 1050Ti 上，使用 paraformer-zh 模型，处理 46 分钟音频大约需要 3~6 分钟（实时率的 0.1~0.2 倍）。具体取决于音频复杂度。
