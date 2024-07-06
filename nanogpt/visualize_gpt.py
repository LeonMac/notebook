import gpt
import sys
import os
import torch


from torchviz import make_dot # for model visualize



def visualize_torchviz(save_path:str, mdl_level:str):
    save_format = 'png'
    print(f'prepare saving net structure to {save_path}.{save_format}')
    # m, xb, yb = gpt.gen_gpt_data_model_for_visualization(mdl_level)
    m      = gpt.gen_model(mdl_level, gpt.device)
    xb, yb = gpt.gen_data(mdl_level, gpt.device)
    if mdl_level in['head','multihead', 'multilayer']:
        make_dot(yb, params=dict(list(m.named_parameters()))).render(save_path, format=save_format)
    elif mdl_level == 'gpt': 
        _, loss = m(xb, yb)
        make_dot(loss, params=dict(list(m.named_parameters()))).render(save_path, format=save_format)
    else:
        exit(0)


def visualize_netron(save_path:str, mdl_level:str):
    # import onnx
    save_file = f"{save_path}.onnx"
    print(f'prepare saving net structure to {save_file}')
    m      = gpt.gen_model(mdl_level, gpt.device)
    xb, yb = gpt.gen_data(mdl_level, gpt.device)


    if mdl_level in['head','multihead', 'multilayer']:
        torch.onnx.export(m, xb, save_file)

    elif mdl_level == 'gpt': 
        torch.onnx.export(m, (xb, yb), save_file)

    else:
        exit(0)



if __name__ == "__main__":
    # 检查是否有足够的参数
    if len(sys.argv) == 3:

        complex_model = sys.argv[1]
        mdl_name = sys.argv[2]

    else:
        print("没有提供足够合适命令行参数。")
        exit(0)

    # DRY_RUN = True if dry_run == 'yes' else False
    BIG     = True if complex_model == 'big' else False

    if mdl_name == 'all':
        for iter_name in gpt.llm_level:
            gpt.global_cofig(iter_name, BIG)
            nn_save_path = os.path.join(gpt.prj_path, gpt.model_arch_dir, gpt.save_nn_name)
            visualize_torchviz(nn_save_path, iter_name)
            visualize_netron(nn_save_path, iter_name)
    
    else:
        gpt.global_cofig(mdl_name, BIG)
        nn_save_path = os.path.join(gpt.prj_path, gpt.model_arch_dir, gpt.save_nn_name)
        visualize_torchviz(nn_save_path, mdl_name)
        visualize_netron(nn_save_path, mdl_name)
    
