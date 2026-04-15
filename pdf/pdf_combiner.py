import os
from io import BytesIO
from pypdf import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch

def create_page_number_pdf(page_num, page_width, page_height):
    """
    生成一个只包含指定页码的 PDF 页面（BytesIO 对象）。
    :param page_num: 页码数字
    :param page_width: 原始页面的宽度（points）
    :param page_height: 原始页面的高度（points）
    :return: BytesIO 对象，可作为 PDF 读取
    """
    packet = BytesIO()
    # 创建与原始页面相同尺寸的画布
    c = canvas.Canvas(packet, pagesize=(page_width, page_height))
    # 设置字体和大小（根据页面大小适当调整）
    font_size = 12
    c.setFont("Helvetica", font_size)
    # 计算页码文本的宽度，用于居中
    text = str(page_num)
    text_width = c.stringWidth(text, "Helvetica", font_size)
    # 页码位置：底部居中，距离下边界 0.5 英寸
    x = (page_width - text_width) / 2
    y = 0.5 * inch
    c.drawString(x, y, text)
    c.save()
    packet.seek(0)
    return packet

def merge_pdfs_with_page_numbers(directory="."):
    """
    合并目录下所有 PDF 文件，并添加连续页码。
    :param directory: 要扫描的目录路径（默认当前目录）
    """
    # 获取目录下所有 PDF 文件（不递归子目录）
    pdf_files = [f for f in os.listdir(directory) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print("当前目录下没有找到 PDF 文件。")
        return

    # 按文件名排序（确保合并顺序稳定）
    pdf_files.sort()

    # 准备 PDF 写入器
    writer = PdfWriter()
    page_counter = 1  # 全局页码从 1 开始

    for filename in pdf_files:
        filepath = os.path.join(directory, filename)
        print(f"正在处理: {filename}")
        try:
            reader = PdfReader(filepath)
            for page in reader.pages:
                # 获取原始页面的尺寸
                # 使用 .mediabox 获取页面尺寸（若没有则使用 .artbox 或 .cropbox）
                page_box = page.mediabox
                page_width = float(page_box.width)
                page_height = float(page_box.height)

                # 生成当前页码的水印 PDF
                watermark_packet = create_page_number_pdf(page_counter, page_width, page_height)
                watermark_reader = PdfReader(watermark_packet)
                watermark_page = watermark_reader.pages[0]

                # 将水印合并到原始页面
                page.merge_page(watermark_page)
                # 将合并后的页面添加到最终 PDF
                writer.add_page(page)

                page_counter += 1
        except Exception as e:
            print(f"处理文件 {filename} 时出错: {e}")

    # 写入最终合并文件
    output_path = os.path.join(directory, "combine.pdf")
    with open(output_path, "wb") as f_out:
        writer.write(f_out)

    print(f"合并完成，共处理 {page_counter - 1} 页。")
    print(f"输出文件: {output_path}")

if __name__ == "__main__":
    # 可以修改为你的目标目录，例如 merge_pdfs_with_page_numbers("/path/to/your/dir")
    merge_pdfs_with_page_numbers(".")
