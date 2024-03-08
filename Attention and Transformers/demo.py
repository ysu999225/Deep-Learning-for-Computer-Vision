from datasets import get_cifar
import torch, os
from tensorboardX import SummaryWriter
from finetune import Trainer, test, ViTLinear, inference
from torch.utils.data import DataLoader
import yaml

exp_name = 'vit_linear'
encoder = 'vit_b_32'

def get_config(exp_name, encoder):
    dir_name = f'runs-1-{encoder}-cifar100/demo-{exp_name}'
    
    encoder_registry = {
        'ViTLinear': ViTLinear,
    }
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)[exp_name]

    lr = config['lr']
    wd = config['wd']
    epochs = config['epochs']
    optimizer = config['optimizer']
    scheduler = config['scheduler']
    momentum = config['momentum']
    net_class = encoder_registry[config['net_class']]

    return net_class, dir_name, (optimizer, lr, wd, momentum), (scheduler, epochs)

def worker(exp_name, encoder):
    torch.set_num_threads(2)
    torch.manual_seed(0)

    batch_size = 32
    
    net_class, dir_name, \
        (optimizer, lr, wd, momentum), \
        (scheduler, epochs) = \
            get_config(exp_name, encoder)
    
    tmp_file_name = dir_name + '/best_model.pth'
    device = torch.device('cuda:0')
    
    
    writer = SummaryWriter(f'{dir_name}/lr{lr:0.6f}_wd{wd:0.6f}', flush_secs=10)

    train_dataset, val_dataset= get_cifar('train'), get_cifar('val')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Prepare the model
    model = net_class(len(train_dataset.classes), encoder)
    trainer = Trainer(model, train_loader, val_loader, writer, 
                        optimizer=optimizer, lr=lr, wd=wd, momentum=momentum,
                        scheduler=scheduler, epochs=epochs,
                        device=device)
    best_val_acc, best_epoch = trainer.train(model_file_name=tmp_file_name)
    print(f"lr: {lr:0.7f}, wd: {wd:0.7f}, best_val_acc: {best_val_acc}, best_epoch: {best_epoch}")
    print()
    model, trainer, train_loader, val_loader, train_dataset, val_dataset = \
        None, None, None, None, None, None
    torch.cuda.empty_cache()

    # Confirm we get the same results on the val set.
    print('Confirm we get the same results on the val set.')
    val_dataset = get_cifar('val')
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    model = net_class(len(val_dataset.classes), encoder)
    model.load_state_dict(torch.load(tmp_file_name))
    model.to(device)
    val_loss, val_accuracy = test(val_loader, model, device)
    print(f"val_loss: {val_loss}, val_accuracy: {val_accuracy}")
    print('')

    # Save predictions on the test set.
    print('Getting predictions on the test set for the best model.')
    test_dataset = get_cifar('test')
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    inference(test_loader, model, device, 
              result_path=os.path.join(dir_name, 'test_predictions.txt'))

def main(_):
    worker(exp_name, encoder)

if __name__ == '__main__':
        main(None)
