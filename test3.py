from yolov8_corn import prune_yolov8

if __name__ == '__main__':
    prune_yolov8(
        model_path='yolov8n.pt',
        data_yaml=r'/gemini/code/Torch-Pruning-master/examples/yolov8/data.yaml',
        sparsity_rate=0.02,
        prune_ratio=0.25,
        sparse_epochs=100,
        finetune_epochs=50,
        batch=16,
        imgsz=160,
        device='0',        # GPU; 改为 'cpu' 用 CPU
        do_distill=True,
    )
