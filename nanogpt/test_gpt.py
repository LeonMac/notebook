import gpt
import sys



if __name__ == "__main__":
    # 检查是否有足够的参数
    if len(sys.argv) == 4:
        print_iter = int(sys.argv[1])
        dry_run = sys.argv[2]
        complex_model = sys.argv[3]

    else:
        print("没有提供足够合适命令行参数。")
        exit(0)

    DRY_RUN = True if dry_run == 'yes' else False
    BIG     = True if complex_model == 'big' else False

    # if complex_model

    gpt.global_cofig(None,BIG)

    
    model_name_list = ['first','second','third','fourth','fifth']
    iter_list       = [100,    100,    100,   100,   100]

    # tc.memory._record_memory_history()

    for n in range(len(model_name_list)):

        if n == 0:
            load_name = None
        else:
            load_name = model_name_list[n-1]

        save_name = model_name_list[n]

        # print(f"n={n}, load_name = {load_name}, save_name={save_name}")

        gpt.train_model(iter_list[n], print_iter, load_name, save_name,  DRY_RUN)

        gpt.test_model(save_name, iter_list[n], 300)

    # tc.memory._dump_snapshot("my_snapshot.pickle")


