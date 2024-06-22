import gpt
import sys
import os
import torch


from torchviz import make_dot # for model visualize



def visualize_torchviz(save_name:str):
    m, xb, yb = gpt.gen_data_model_for_visualize()
    _, loss = m(xb, yb)
    save_format = 'png'
    print(f'saving net structure to {save_name}.{save_format}')
    make_dot(loss, params=dict(list(m.named_parameters()))).render(save_name, format=save_format)

def visualize_netron(save_name:str):
    # import onnx
    save_file = f"{save_name}.onnx"
    print(f'saving net structure to {save_file}')
    m, xb, yb = gpt.gen_data_model_for_visualize()
    onnx_program = torch.onnx.export(m, (xb, yb), save_file)
    # onnx_program.save(save_file)

if __name__ == "__main__":
    # 检查是否有足够的参数
    if len(sys.argv) == 2:

        complex_model = sys.argv[1]

    else:
        print("没有提供足够合适命令行参数。")
        exit(0)

    # DRY_RUN = True if dry_run == 'yes' else False
    BIG     = True if complex_model == 'big' else False


    gpt.global_cofig(BIG)

    nn_save_path = os.path.join(gpt.prj_path, gpt.model_arch_dir, gpt.save_nn_name)
    visualize_torchviz(nn_save_path)
    visualize_netron(nn_save_path)
    
